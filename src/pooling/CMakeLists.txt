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
  if(${DATA_FORMAT} STREQUAL NHWC OR (VECTOR_WIDTH EQUAL 1 AND
      ${dir} STREQUAL Forward))
    string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
    set(_filename "${INST_POOL_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${op}")
    set(_filename "${_filename}_${dir}_${VECTOR_WIDTH}_${USE_FASTDIV}")
    set(_filename "${_filename}_${DATA_FORMAT}.cc")
    set(_gen_file ${CMAKE_BINARY_DIR}/generated/pooling/${_filename})
    set(OPERATOR ${op})
    set(DIRECTION ${dir})
    configure_file(${template} ${_gen_file} @ONLY)
    list(APPEND ${out_var} ${_gen_file})
  endif()
endmacro()

function(generate_pooling)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_POOL
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_general_template queue_pooling_kernel_impl.cc.in)
  set(_max_grad_template queue_max_grad_kernel_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
        foreach(USE_FASTDIV IN ITEMS true false)
          foreach(DATA_FORMAT IN LISTS SNN_LAYOUTS)
            generate_kernel(_sources ${_general_template} Average Forward)
            generate_kernel(_sources ${_general_template} Average Backpropagate)
            generate_kernel(_sources ${_general_template} Max Forward)
            generate_kernel(_sources ${_general_template} MaxWithNan Forward)
            generate_kernel(_sources ${_max_grad_template} Max Backpropagate)
            generate_kernel(_sources ${_max_grad_template} MaxWithNan Backpropagate)
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  set(${INST_POOL_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_pooling(
  OUTPUT_VAR pooling_kernels
  FILENAME   pool
)

snn_object_library(
  WITH_SYCL
  TARGET pooling
  KERNEL_SOURCES
    ${pooling_kernels}
  SOURCES
    launch_pooling.cc
    launch_max_grad_pooling.cc
)
