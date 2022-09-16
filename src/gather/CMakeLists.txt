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
include(SNNHelpers)

macro(generate_gather_impl out_var template)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${GEN_GATHER_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}")
  set(_filename "${_filename}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/gather/${_filename})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

set(GATHER_INDICES_TYPES int32_t int64_t)

function(generate_gather_kernels)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(GEN_GATHER
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_INT_TYPES)
    foreach(INDEX_TYPE IN LISTS GATHER_INDICES_TYPES)
        generate_gather_impl(_sources queue_kernel_impl.cc.in)
    endforeach()
  endforeach()
  set(${GEN_GATHER_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_gather_kernels(
  OUTPUT_VAR    gather_kernels
  FILENAME      gather_kernel
)
snn_object_library(
  WITH_SYCL
  TARGET gather
  KERNEL_SOURCES 
    ${gather_kernels}
  SOURCES        
    launch.cc
)
