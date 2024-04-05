# Copyright Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10.2)
project(snn_bench)
include(HandleBenchmark)
include(SNNHelpers)

# Creates a git-config.tmp file containing the commit hash and date. A custom
# target ensures that this command is always executed.
add_custom_target(generate_git_config
  ${CMAKE_CURRENT_SOURCE_DIR}/make_git_config.$<IF:$<BOOL:${WIN32}>,bat,sh> >
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

# Assume ComputeCpp not available by default.
set(ComputeCpp_INFO_AVAILABLE false)
set(ComputeCpp_VERSION_NUMBER "N/A")
set(ComputeCpp_EDITION "N/A")

if(ComputeCpp_FOUND)
  execute_process(COMMAND ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE} "--version"
    OUTPUT_VARIABLE ComputeCpp_DEVICE_COMPILER_VERSION
    RESULT_VARIABLE ComputeCpp_DEVICE_COMPILER_EXECUTABLE_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT ComputeCpp_DEVICE_COMPILER_EXECUTABLE_RESULT EQUAL "0")
    message(WARNING "Compute++ not found - Error obtaining device compiler and ComputeCpp version!")
  else()
    # Store information about ComputeCpp/compiler for benchmarking.
    set(ComputeCpp_INFO_AVAILABLE true)
    string(REGEX MATCH
      "(CE|PE|RC)" ComputeCpp_EDITION ${ComputeCpp_DEVICE_COMPILER_VERSION})

    if(${ComputeCpp_EDITION} STREQUAL "RC")
      set(ComputeCpp_EDITION "Internal")
    endif()

    string(REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)"
      ComputeCpp_VERSION_NUMBER ${ComputeCpp_DEVICE_COMPILER_VERSION})
  endif()
endif()

configure_file(computecpp_version_config.h.in computecpp_version_config.h @ONLY)

add_library(bench_info STATIC bench_info.cc git_config.h computecpp_version_config.h)
target_include_directories(bench_info PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_library(bench_main STATIC bench_main.cc)
target_link_libraries(bench_main PUBLIC benchmark::benchmark bench_info)

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

if(SNN_BENCH_PORTBLAS)
  include(HandlePortBLAS)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_BENCH_PORTBLAS)
  list(APPEND _BACKENDS PORTBLAS::portblas)
endif()

if(SNN_BENCH_CLBLAST)
  include(HandleCLBlast)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_BENCH_CLBLAST)
  list(APPEND _BACKENDS clblast)
endif()

if(SNN_BENCH_SNN)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_BENCH_SNNBACKEND)
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

snn_object_library(
  TARGET
  simple_conv2d_config
  SOURCES
  conv2d/simple_convolution.cc
  PUBLIC_LIBRARIES
  benchmark::benchmark
  PUBLIC_COMPILE_DEFINITIONS
  ${_BENCHMARK_DEFINITIONS}
)

function (snn_compile_benchmark_functions operator)
  snn_object_library(
    WITH_SYCL
    TARGET
    ${operator}_benchmark_functions
    KERNEL_SOURCES
    ${operator}/benchmark_functions.cc
    PUBLIC_LIBRARIES
    benchmark::benchmark
    ${_BACKENDS}
    PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
  )
endfunction()

snn_compile_benchmark_functions(conv2d)
snn_compile_benchmark_functions(pooling)
snn_compile_benchmark_functions(batchnorm)
snn_compile_benchmark_functions(depthwise_conv2d)
snn_compile_benchmark_functions(bias)
snn_compile_benchmark_functions(binary)

function(snn_operator_bench operator modelname)
  snn_object_library(
    TARGET
    ${modelname}_${operator}_config
    SOURCES
    ${operator}/${modelname}.cc
    PUBLIC_LIBRARIES
    benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
  )
  snn_bench(
    WITH_SYCL
    TARGET
    ${modelname}_${operator}
    OBJECTS
    $<TARGET_OBJECTS:${operator}_benchmark_functions>
    $<TARGET_OBJECTS:${modelname}_${operator}_config>
    PUBLIC_LIBRARIES
    bench_main
    sycl_dnn
    ${_BACKENDS}
  )
endfunction()

snn_operator_bench(conv2d mobilenet)
snn_operator_bench(conv2d ssd_mobilenet)
snn_operator_bench(conv2d resnet)
snn_operator_bench(conv2d vgg)
snn_operator_bench(conv2d xception)

snn_operator_bench(pooling mobilenet)
snn_operator_bench(pooling resnet)
snn_operator_bench(pooling vgg)

snn_operator_bench(batchnorm resnet50)

snn_operator_bench(depthwise_conv2d mobilenet)
snn_operator_bench(depthwise_conv2d xception)

snn_operator_bench(bias net)

snn_operator_bench(binary net_bias)

snn_bench(
  WITH_SYCL
  TARGET
    binary
  SOURCES
    binary/binary.cc
  PUBLIC_LIBRARIES
    bench_main
    sycl_dnn
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

snn_bench(
  WITH_SYCL
  TARGET
  relu
  SOURCES
  pointwise/relu.cc
  PUBLIC_LIBRARIES
  bench_main
  sycl_dnn
  PUBLIC_COMPILE_DEFINITIONS
  ${_BENCHMARK_DEFINITIONS}
)

snn_bench(
  WITH_SYCL
  TARGET
  tanh
  SOURCES
  pointwise/tanh.cc
  PUBLIC_LIBRARIES
  bench_main
  sycl_dnn
  PUBLIC_COMPILE_DEFINITIONS
  ${_BENCHMARK_DEFINITIONS}
)

snn_bench(
  WITH_SYCL
  TARGET
  concat
  SOURCES
  concat/concat.cc
  PUBLIC_LIBRARIES
  bench_main
  sycl_dnn
  PUBLIC_COMPILE_DEFINITIONS
  ${_BENCHMARK_DEFINITIONS}
)

snn_bench(
  WITH_SYCL
  TARGET
    reduction_benchmark_functions
  SOURCES
    reduction/benchmark_functions.cc
    reduction/reductions.cc
  PUBLIC_LIBRARIES
    bench_main
    sycl_dnn
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

snn_bench(
  WITH_SYCL
  TARGET
    nd_reduction_benchmark_functions
  SOURCES
    nd_reduction/benchmark_functions.cc
    nd_reduction/nd_reductions.cc
  PUBLIC_LIBRARIES
    bench_main
    sycl_dnn
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

add_subdirectory(matmul)

if(SNN_BUILD_INTERNAL_BENCHMARKS)
  add_subdirectory(internal)
endif()

if(SNN_BENCH_ARM_COMPUTE)
  find_package(ARMCompute REQUIRED)
  function(snn_arm_compile_benchmark_functions operator)
    snn_object_library(
      TARGET
      arm_neon_${operator}_benchmark_functions
      SOURCES
      ${operator}/arm_benchmark_functions.cc
      PUBLIC_LIBRARIES
      benchmark::benchmark
      ARMCompute::ARMCompute
      ARMCompute::Core
      PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
    )
    snn_object_library(
      TARGET
      arm_opencl_${operator}_benchmark_functions
      SOURCES
      ${operator}/arm_benchmark_functions.cc
      PUBLIC_LIBRARIES
      benchmark::benchmark
      ARMCompute::ARMCompute
      ARMCompute::Core
      PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
    )
  endfunction()
  snn_arm_compile_benchmark_functions(conv2d)
  snn_arm_compile_benchmark_functions(depthwise_conv2d)
  
  function(snn_arm_conv2d_bench operator modelname)
    snn_bench(
      TARGET
      arm_opencl_${modelname}_${operator}
      OBJECTS
      $<TARGET_OBJECTS:arm_opencl_${operator}_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_${operator}_config>
      PUBLIC_LIBRARIES
      bench_main
      ARMCompute::ARMCompute
      ARMCompute::Core
    )
    snn_bench(
      TARGET
      arm_neon_${modelname}_${operator}
      OBJECTS
      $<TARGET_OBJECTS:arm_neon_${operator}_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_${operator}_config>
      PUBLIC_LIBRARIES
      bench_main
      ARMCompute::ARMCompute
      ARMCompute::Core
    )
  endfunction()

  snn_arm_bench(conv2d mobilenet)
  snn_arm_bench(conv2d resnet)
  snn_arm_bench(conv2d ssd_mobilenet)
  snn_arm_bench(conv2d vgg)
  snn_arm_bench(conv2d xception)

  snn_arm_bench(depthwise_conv2d mobilenet)
  snn_arm_bench(depthwise_conv2d xception)
endif()

if(SNN_BENCH_MKLDNN)
  find_package(mkldnn REQUIRED)
  function (snn_mkl_compile_benchmark_functions operator)
    snn_object_library(
      TARGET
      mkl_${operator}_benchmark_functions
      SOURCES
      ${operator}/mkl_benchmark_functions.cc
      PUBLIC_LIBRARIES
      benchmark::benchmark
      MKLDNN::mkldnn
      PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
    )
  endfunction()

  snn_mkl_compile_benchmark_functions(conv2d)
  snn_mkl_compile_benchmark_functions(depthwise_conv2d)

  function(snn_mkl_bench modelname)
    snn_bench(
      TARGET
      mkl_${modelname}_${operator}
      OBJECTS
      $<TARGET_OBJECTS:mkl_${operator}_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_${operator}_config>
      PUBLIC_LIBRARIES
      bench_main
      MKLDNN::mkldnn
    )
  endfunction()

  snn_mkl_bench(conv2d mobilenet)
  snn_mkl_bench(conv2d resnet)
  snn_mkl_bench(conv2d ssd_mobilenet)
  snn_mkl_bench(conv2d vgg)
  snn_mkl_bench(conv2d xception)

  snn_mkl_bench(depthwise_conv2d mobilenet)
  snn_mkl_bench(depthwise_conv2d xception)
endif()

if(SNN_BENCH_CUDNN)
  find_package(CUDAToolkit REQUIRED)
  find_library(CUDNN_LIBRARY
    NAMES libcudnn.so
    DOC "The cuDNN library"
  )
  add_library(cudnn SHARED IMPORTED)
  set_target_properties(cudnn PROPERTIES IMPORTED_LOCATION ${CUDNN_LIBRARY})

  function(snn_cudnn_compile_benchmark_functions operator)
    snn_object_library(
      WITH_SYCL
      TARGET
      cudnn_${operator}_benchmark_functions
      SOURCES
      ${operator}/cudnn_benchmark_functions.cc
      PUBLIC_LIBRARIES
      benchmark::benchmark
      CUDA::cudart
      CUDA::cuda_driver
      sycl_dnn
      PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
    )
  endfunction()
  snn_cudnn_compile_benchmark_functions(conv2d)
  snn_cudnn_compile_benchmark_functions(batchnorm)
  snn_cudnn_compile_benchmark_functions(pooling)

  function(snn_cudnn_bench operator modelname)
    snn_bench(
      WITH_SYCL
      TARGET
      cudnn_${modelname}_${operator}
      OBJECTS
      $<TARGET_OBJECTS:cudnn_${operator}_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_${operator}_config>
      PUBLIC_LIBRARIES
      CUDA::cudart
      CUDA::cuda_driver
      cudnn
      bench_main
      sycl_dnn
    )
  endfunction()

  snn_cudnn_bench(conv2d mobilenet)
  snn_cudnn_bench(conv2d resnet)
  snn_cudnn_bench(conv2d vgg)

  snn_cudnn_bench(batchnorm resnet50)

  snn_cudnn_bench(pooling mobilenet)
  snn_cudnn_bench(pooling resnet)
  snn_cudnn_bench(pooling vgg)
endif()

if(SNN_BENCH_MIOPEN)
  set(ROCM_PATH "/opt/rocm" CACHE STRING 
      "Path to rocm installation. /opt/rocm by default")
  list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/hip/cmake")
  find_package(hip REQUIRED)
  find_package(miopen REQUIRED)
  target_compile_definitions(hip::amdhip64 INTERFACE __HIP_PLATFORM_AMD__)

  function(snn_miopen_compile_benchmark_functions operator)
    snn_object_library(
      WITH_SYCL
      TARGET
      miopen_${operator}_benchmark_functions
      SOURCES
      ${operator}/miopen_benchmark_functions.cc
      PUBLIC_LIBRARIES
      benchmark::benchmark
      hip::amdhip64
      sycl_dnn
      PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
    )
  endfunction()
  snn_miopen_compile_benchmark_functions(conv2d)
  snn_miopen_compile_benchmark_functions(batchnorm)
  snn_miopen_compile_benchmark_functions(pooling)

  function(snn_miopen_bench operator modelname)
    snn_bench(
      WITH_SYCL
      TARGET
      miopen_${modelname}_${operator}
      OBJECTS
      $<TARGET_OBJECTS:miopen_${operator}_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_${operator}_config>
      PUBLIC_LIBRARIES
      hip::amdhip64
      MIOpen
      bench_main
      sycl_dnn
    )
  endfunction()

  snn_miopen_bench(conv2d mobilenet)
  snn_miopen_bench(conv2d resnet)
  snn_miopen_bench(conv2d vgg)

  snn_miopen_bench(batchnorm resnet50)
  
  snn_miopen_bench(pooling mobilenet)
  snn_miopen_bench(pooling resnet)
  snn_miopen_bench(pooling vgg)
endif()
