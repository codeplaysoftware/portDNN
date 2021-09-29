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

add_definitions(-DENABLE_SOFTMAXDIV=1)

macro(generate_kernel out_var template op dir)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_SOFTMAX_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${op}")
  set(_filename "${_filename}_${dir}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/softmax/${_filename})
  set(OP_TYPE ${op})
  set(DIRECTION ${dir})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_softmax)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_SOFTMAX
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_forward_template ../pointwise/queue_pointwise_forward_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
        generate_kernel(_sources ${_forward_template} Exp Forward)
        generate_kernel(_sources ${_forward_template} SoftMaxDiv Forward)
      endforeach()
    endforeach()
  endforeach()
  set(${INST_SOFTMAX_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_softmax(
  OUTPUT_VAR softmax_kernels
  FILENAME   softmax
)

snn_object_library(
  WITH_SYCL
  TARGET 
    softmax
  KERNEL_SOURCES
    ${softmax_kernels}
  SOURCES
    ../pointwise/launch_pointwise_forward.cc
)
set(-DENABLE_SOFTMAXDIV 0)
