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

macro(generate_kernel out_var template) 
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_BIAS_FILENAME}_${DTYPE_ID}")
  set(_filename "${_filename}_${dir}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/bias/${_filename})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_bias)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_BIAS
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_general_template queue_bias_kernel_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
    foreach(VECTOR_WIDTH IN ITEMS 1 2 4 8)
        generate_kernel(_sources ${_general_template})
      endforeach()
    endforeach()
  endforeach()
  set(${INST_BIAS_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_bias(
  OUTPUT_VAR bias_kernels
  FILENAME   biasadd
)

snn_object_library(
  WITH_SYCL
  TARGET bias
  KERNEL_SOURCES
    ${bias_kernels}
  SOURCES
    launch_bias_add.cc
)
