# Copyright 2018 Codeplay Software Ltd.
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

cmake_minimum_required(VERSION 3.2.2)
include(SNNHelpers)

macro(generate_matmul_impl out_var row_tile acc_tile col_tile)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${GEN_MATMUL_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}")
  set(_filename "${_filename}_${row_tile}_${acc_tile}_${col_tile}")
  set(_filename "${_filename}_${TRANS_LHS}_${TRANS_RHS}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/matmul/${_filename})
  set(ROW_TILE ${row_tile})
  set(ACC_TILE ${acc_tile})
  set(COL_TILE ${col_tile})
  configure_file(${GEN_MATMUL_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_matmul_kernels)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(GEN_MATMUL
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_sources "")
  set(_bool_list true false)
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(TRANS_LHS IN LISTS _bool_list)
        foreach(TRANS_RHS IN LISTS _bool_list)
          generate_matmul_impl(_sources 4 4 4)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  set(${GEN_MATMUL_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_matmul_kernels(
  OUTPUT_VAR    matmul_kernel_sources
  TEMPLATE_FILE queue_kernel_impl.cc.in
  FILENAME      matmul_kernel
)
snn_object_library(
  WITH_SYCL
  TARGET         matmul
  SOURCES        launch.cc
  KERNEL_SOURCES ${matmul_kernel_sources}
)
