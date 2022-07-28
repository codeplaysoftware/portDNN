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

macro(generate_transpose_impl out_var n_dim)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${GEN_TRANSPOSE_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}")
  set(_filename "${_filename}_${n_dim}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/transpose/${_filename})
  set(N_DIM ${n_dim})
  configure_file(${GEN_TRANSPOSE_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_transpose_kernels)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(GEN_TRANSPOSE
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_INT_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      generate_transpose_impl(_sources 2)
      generate_transpose_impl(_sources 3)
      generate_transpose_impl(_sources 4)
      generate_transpose_impl(_sources 5)
      generate_transpose_impl(_sources 6)
    endforeach()
  endforeach()
  set(${GEN_TRANSPOSE_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_transpose_kernels(
  OUTPUT_VAR    transpose_kernel_sources
  TEMPLATE_FILE queue_kernel_impl.cc.in
  FILENAME      transpose_kernel
)
snn_object_library(
  WITH_SYCL
  TARGET         transpose
  SOURCES        launch.cc
  KERNEL_SOURCES ${transpose_kernel_sources}
)

