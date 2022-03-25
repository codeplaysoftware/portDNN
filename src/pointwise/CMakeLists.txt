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

cmake_minimum_required(VERSION 3.10.2)
include(SNNHelpers)

macro(generate_kernel out_var template op dir)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_PWISE_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${op}")
  set(_filename "${_filename}_${dir}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/pointwise/${_filename})
  set(OP_TYPE ${op})
  set(DIRECTION ${dir})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_pointwise)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_PWISE
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_forward_template queue_pointwise_forward_impl.cc.in)
  set(_grad_template queue_pointwise_grad_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
        generate_kernel(_sources ${_forward_template} Relu Forward)
        generate_kernel(_sources ${_grad_template} Relu Gradient)
        generate_kernel(_sources ${_forward_template} Tanh Forward)
        generate_kernel(_sources ${_grad_template} Tanh Gradient)
        generate_kernel(_sources ${_forward_template} Exp Forward)
        generate_kernel(_sources ${_grad_template} Exp Gradient)
        generate_kernel(_sources ${_forward_template} Log Forward)
        generate_kernel(_sources ${_grad_template} Log Gradient)
        generate_kernel(_sources ${_forward_template} Floor Forward)
        generate_kernel(_sources ${_forward_template} Sqrt Forward)
        generate_kernel(_sources ${_grad_template} Sqrt Gradient)
      endforeach()
    endforeach()
  endforeach()
  set(${INST_PWISE_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_pointwise(
  OUTPUT_VAR pointwise_kernels
  FILENAME   pointwise
)
snn_object_library(
  WITH_SYCL
  TARGET pointwise
  KERNEL_SOURCES
    ${pointwise_kernels}
  SOURCES
    launch_pointwise_forward.cc
    launch_pointwise_grad.cc
)
