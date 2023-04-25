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

if(SNN_BENCH_SYCLBLAS)
  include(Handlesycl_blas)
  list(APPEND _BENCHMARK_DEFINITIONS -DSNN_BENCH_SYCLBLAS)
  list(APPEND _BACKENDS SYCL_BLAS::sycl_blas)
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
  WITH_SYCL
  TARGET
    conv2d_benchmark_functions
  KERNEL_SOURCES
    conv2d/benchmark_functions.cc
  PUBLIC_LIBRARIES
    benchmark::benchmark
    ${_BACKENDS}
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)
snn_object_library(
  TARGET
    simple_convolution_config
  SOURCES
    conv2d/simple_convolution.cc
  PUBLIC_LIBRARIES
    benchmark::benchmark
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

function(snn_conv2d_config_lib modelname)
  snn_object_library(
    TARGET
      ${modelname}_convolution_config
    SOURCES
      conv2d/${modelname}.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
endfunction()

snn_conv2d_config_lib(mobilenet)
snn_conv2d_config_lib(ssd_mobilenet)
snn_conv2d_config_lib(resnet)
snn_conv2d_config_lib(vgg)
snn_conv2d_config_lib(xception)

function(snn_conv2d_bench modelname)
  snn_bench(
    WITH_SYCL
    TARGET
      ${modelname}_convolution
    OBJECTS
      $<TARGET_OBJECTS:conv2d_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_convolution_config>
    PUBLIC_LIBRARIES
      bench_main
      sycl_dnn
      ${_BACKENDS}
  )
endfunction()

snn_conv2d_bench(simple)
snn_conv2d_bench(mobilenet)
snn_conv2d_bench(ssd_mobilenet)
snn_conv2d_bench(resnet)
snn_conv2d_bench(vgg)
snn_conv2d_bench(xception)

snn_object_library(
  WITH_SYCL
  TARGET
    pooling_benchmark_functions
  KERNEL_SOURCES
    pooling/benchmark_functions.cc
  PUBLIC_LIBRARIES
    benchmark::benchmark
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

function(snn_pooling_config_lib modelname)
  snn_object_library(
    TARGET
      ${modelname}_pooling_config
    SOURCES
      pooling/${modelname}.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
endfunction()

snn_pooling_config_lib(mobilenet)
snn_pooling_config_lib(resnet)
snn_pooling_config_lib(vgg)

function(snn_pooling_bench modelname)
  snn_bench(
    WITH_SYCL
    TARGET
      ${modelname}_pooling
    OBJECTS
      $<TARGET_OBJECTS:pooling_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_pooling_config>
    PUBLIC_LIBRARIES
      bench_main
      sycl_dnn
  )
endfunction()

snn_pooling_bench(mobilenet)
snn_pooling_bench(resnet)
snn_pooling_bench(vgg)

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

snn_object_library(
  WITH_SYCL
  TARGET
    depthwise_conv2d_benchmark_functions
  KERNEL_SOURCES
    depthwise_conv2d/benchmark_functions.cc
  PUBLIC_LIBRARIES
    benchmark::benchmark
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

function(snn_depthwise_conv2d_config_lib modelname)
  snn_object_library(
    TARGET
      ${modelname}_depthwise_conv2d_config
    SOURCES
      depthwise_conv2d/${modelname}.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
endfunction()

snn_depthwise_conv2d_config_lib(mobilenet)
snn_depthwise_conv2d_config_lib(xception)

function(snn_depthwise_conv2d_bench modelname)
  snn_bench(
    WITH_SYCL
    TARGET
      ${modelname}_depthwise_convolution
    OBJECTS
      $<TARGET_OBJECTS:depthwise_conv2d_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_depthwise_conv2d_config>
    PUBLIC_LIBRARIES
      bench_main
      sycl_dnn
  )
endfunction()

snn_depthwise_conv2d_bench(mobilenet)
snn_depthwise_conv2d_bench(xception)

add_subdirectory(matmul)

if(SNN_BUILD_INTERNAL_BENCHMARKS)
  add_subdirectory(internal)
endif()

if(SNN_BENCH_ARM_COMPUTE)
  find_package(ARMCompute REQUIRED)
  snn_object_library(
    TARGET
      arm_opencl_conv2d_benchmark_functions
    SOURCES
      conv2d/arm_benchmark_functions.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
      ARMCompute::ARMCompute
      ARMCompute::Core
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
  snn_object_library(
    TARGET
      arm_neon_conv2d_benchmark_functions
    SOURCES
      conv2d/arm_benchmark_functions.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
      ARMCompute::ARMCompute
      ARMCompute::Core
    PUBLIC_COMPILE_DEFINITIONS
      ARM_NEON
      ${_BENCHMARK_DEFINITIONS}
  )

  function(snn_arm_conv2d_bench modelname)
    snn_bench(
      TARGET
        arm_opencl_${modelname}_convolution
      OBJECTS
        $<TARGET_OBJECTS:arm_opencl_conv2d_benchmark_functions>
        $<TARGET_OBJECTS:${modelname}_convolution_config>
      PUBLIC_LIBRARIES
        bench_main
        ARMCompute::ARMCompute
        ARMCompute::Core
    )
    snn_bench(
      TARGET
        arm_neon_${modelname}_convolution
      OBJECTS
        $<TARGET_OBJECTS:arm_neon_conv2d_benchmark_functions>
        $<TARGET_OBJECTS:${modelname}_convolution_config>
      PUBLIC_LIBRARIES
        bench_main
        ARMCompute::ARMCompute
        ARMCompute::Core
    )
  endfunction()
  snn_arm_conv2d_bench(mobilenet)
  snn_arm_conv2d_bench(resnet)
  snn_arm_conv2d_bench(ssd_mobilenet)
  snn_arm_conv2d_bench(vgg)
  snn_arm_conv2d_bench(xception)

  snn_object_library(
    TARGET
      arm_opencl_depthwise_conv2d_benchmark_functions
    SOURCES
      depthwise_conv2d/arm_benchmark_functions.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
      ARMCompute::ARMCompute
      ARMCompute::Core
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
  snn_object_library(
    TARGET
      arm_neon_depthwise_conv2d_benchmark_functions
    SOURCES
      depthwise_conv2d/arm_benchmark_functions.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
      ARMCompute::ARMCompute
      ARMCompute::Core
    PUBLIC_COMPILE_DEFINITIONS
      ARM_NEON
      ${_BENCHMARK_DEFINITIONS}
  )

  function(snn_arm_depthwise_conv2d_bench modelname)
    snn_bench(
      TARGET
        arm_opencl_${modelname}_depthwise_convolution
      OBJECTS
        $<TARGET_OBJECTS:arm_opencl_depthwise_conv2d_benchmark_functions>
        $<TARGET_OBJECTS:${modelname}_depthwise_conv2d_config>
      PUBLIC_LIBRARIES
        bench_main
        ARMCompute::ARMCompute
        ARMCompute::Core
    )
    snn_bench(
      TARGET
        arm_neon_${modelname}_depthwise_convolution
      OBJECTS
        $<TARGET_OBJECTS:arm_neon_depthwise_conv2d_benchmark_functions>
        $<TARGET_OBJECTS:${modelname}_depthwise_conv2d_config>
      PUBLIC_LIBRARIES
        bench_main
        ARMCompute::ARMCompute
        ARMCompute::Core
    )
  endfunction()
  snn_arm_depthwise_conv2d_bench(mobilenet)
  snn_arm_depthwise_conv2d_bench(xception)
endif()

if(SNN_BENCH_MKLDNN)
  find_package(mkldnn REQUIRED)
  snn_object_library(
    TARGET
      mkl_conv2d_benchmark_functions
    SOURCES
      conv2d/mkl_benchmark_functions.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
      MKLDNN::mkldnn
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
  function(snn_mkl_conv2d_bench modelname)
    snn_bench(
      TARGET
        mkl_${modelname}_convolution
      OBJECTS
        $<TARGET_OBJECTS:mkl_conv2d_benchmark_functions>
        $<TARGET_OBJECTS:${modelname}_convolution_config>
      PUBLIC_LIBRARIES
        bench_main
        MKLDNN::mkldnn
    )
  endfunction()
  snn_mkl_conv2d_bench(mobilenet)
  snn_mkl_conv2d_bench(resnet)
  snn_mkl_conv2d_bench(ssd_mobilenet)
  snn_mkl_conv2d_bench(vgg)
  snn_mkl_conv2d_bench(xception)

  snn_object_library(
    TARGET
      mkl_depthwise_conv2d_benchmark_functions
    SOURCES
      depthwise_conv2d/mkl_benchmark_functions.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
      MKLDNN::mkldnn
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
  function(snn_mkl_depthwise_conv2d_bench modelname)
    snn_bench(
      TARGET
        mkl_${modelname}_depthwise_convolution
      OBJECTS
        $<TARGET_OBJECTS:mkl_depthwise_conv2d_benchmark_functions>
        $<TARGET_OBJECTS:${modelname}_depthwise_conv2d_config>
      PUBLIC_LIBRARIES
        bench_main
        MKLDNN::mkldnn
    )
  endfunction()
  snn_mkl_depthwise_conv2d_bench(mobilenet)
  snn_mkl_depthwise_conv2d_bench(xception)
endif()

if(SNN_BENCH_CUDNN)
  find_package(CUDAToolkit REQUIRED)
  find_library(CUDNN_LIBRARY
    NAMES libcudnn.so
    DOC "The cuDNN library"
  )
  add_library(cudnn SHARED IMPORTED)
  set_target_properties(cudnn PROPERTIES IMPORTED_LOCATION ${CUDNN_LIBRARY})
  snn_object_library(
    TARGET
      cudnn_conv2d_benchmark_functions
    SOURCES
      conv2d/cudnn_benchmark_functions.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
      CUDA::cudart
      CUDA::cuda_driver
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
  function(snn_cudnn_conv2d_bench modelname)
    snn_bench(
      TARGET
        cudnn_${modelname}_convolution
      OBJECTS
        $<TARGET_OBJECTS:cudnn_conv2d_benchmark_functions>
        $<TARGET_OBJECTS:${modelname}_convolution_config>
      PUBLIC_LIBRARIES
        CUDA::cudart
        CUDA::cuda_driver
        cudnn
        bench_main
    )
  endfunction()
  snn_cudnn_conv2d_bench(mobilenet)
  snn_cudnn_conv2d_bench(resnet)
  snn_cudnn_conv2d_bench(ssd_mobilenet)
  snn_cudnn_conv2d_bench(vgg)
  snn_cudnn_conv2d_bench(xception)
endif()

snn_object_library(
  WITH_SYCL
  TARGET
    bias_benchmark_functions
  KERNEL_SOURCES
    bias/benchmark_functions.cc
  PUBLIC_LIBRARIES
    benchmark::benchmark
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

function(snn_bias_config_lib modelname)
  snn_object_library(
    TARGET
      ${modelname}_bias_config
    SOURCES
      bias/${modelname}.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
endfunction()

snn_bias_config_lib(net)

function(snn_bias_bench modelname)
  snn_bench(
    WITH_SYCL
    TARGET
      ${modelname}_bias
    OBJECTS
      $<TARGET_OBJECTS:bias_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_bias_config>
    PUBLIC_LIBRARIES
      bench_main
      sycl_dnn
  )
endfunction()

snn_bias_bench(net)

snn_object_library(
  WITH_SYCL
  TARGET
    batchnorm_benchmark_functions
  KERNEL_SOURCES
    batchnorm/benchmark_functions.cc
  PUBLIC_LIBRARIES
    benchmark::benchmark
  PUBLIC_COMPILE_DEFINITIONS
    ${_BENCHMARK_DEFINITIONS}
)

function(snn_batchnorm_config_lib modelname)
  snn_object_library(
    TARGET
      ${modelname}_batchnorm_config
    SOURCES
      batchnorm/${modelname}.cc
    PUBLIC_LIBRARIES
      benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS
      ${_BENCHMARK_DEFINITIONS}
  )
endfunction()

snn_batchnorm_config_lib(resnet50)

function(snn_batchnorm_bench modelname)
  snn_bench(
    WITH_SYCL
    TARGET
      ${modelname}_batchnorm
    OBJECTS
      $<TARGET_OBJECTS:batchnorm_benchmark_functions>
      $<TARGET_OBJECTS:${modelname}_batchnorm_config>
    PUBLIC_LIBRARIES
      bench_main
      sycl_dnn
  )
endfunction()

snn_batchnorm_bench(resnet50)
