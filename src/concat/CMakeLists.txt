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

macro(generate_kernel out_var template)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_CONCAT_FILENAME}_${DTYPE_ID}_${VECTOR_WIDTH}")
  set(_filename "${_filename}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/concat/${_filename})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_concat)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_CONCAT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_concat_template queue_concat_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(VECTOR_WIDTH IN ITEMS 1 2 4 8 16)
      foreach(TARRAY_MAX_SIZE IN ITEMS 6)
        generate_kernel(_sources ${_concat_template})
      endforeach()
    endforeach()
  endforeach()
  set(${INST_CONCAT_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_concat(
  OUTPUT_VAR concat_kernels
  FILENAME   concat
)
snn_object_library(
  WITH_SYCL
  TARGET concat
  KERNEL_SOURCES
    ${concat_kernels}
  SOURCES
    launch_concat.cc
)

