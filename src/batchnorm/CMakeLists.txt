# Copyright Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
include(SNNHelpers)

macro(generate_kernel out_var template operation)
  set(_filename "${INST_BATCHNORM_FILENAME}_${DATA_TYPE}_${INDEX_TYPE}")
  set(_filename "${_filename}_${VECTOR_WIDTH}_${operation}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/batchnorm/${_filename})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_batchnorm)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_BATCHNORM
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_variance_template queue_variance_kernel_impl.cc.in)
  set(_running_mean_variance_template queue_running_mean_variance_kernel_impl.cc.in)
  set(_inference_template queue_batchnorm_kernel_inference_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
        generate_kernel(_sources ${_variance_template} Variance)
        generate_kernel(_sources ${_running_mean_variance_template} RunningMeanVariance)
        generate_kernel(_sources ${_inference_template} Inference)
      endforeach()
    endforeach()
  endforeach()
  set(${INST_BATCHNORM_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_batchnorm(
  OUTPUT_VAR batchnorm_kernels
  FILENAME   batchnorm
)

snn_object_library(
  WITH_SYCL
  TARGET 
    batchnorm
  KERNEL_SOURCES
    ${batchnorm_kernels}
  SOURCES
    launch_variance.cc
    launch_running_mean_variance.cc
    launch_batchnorm_inference.cc
)
