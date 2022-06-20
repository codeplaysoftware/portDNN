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

macro(generate_kernel out_var template kernel_name kernel_extra_args)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(KERNEL_NAME ${kernel_name})
  set(KERNEL_EXTRA_ARGS ${kernel_extra_args})
  set(_filename "${INST_BINARYOP_FILENAME}_${DTYPE_ID}")
  set(_filename "${_filename}_${KERNEL_NAME}_${OP_TYPE}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/binaryop/${_filename})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_binary)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_BINARYOP
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_general_template queue_binaryop_kernel_impl.cc.in)
  set(_sources "")
  set(OPS Add Sub Mul Div)
  set(VEC_KERNELS
    BinaryOpVec
    BinaryOpBcastLhsVec2D
    BinaryOpBcastRhsVec2D
    BinaryOpBcastLhsVec3D
    BinaryOpBcastRhsVec3D
  )
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(OP_TYPE IN LISTS OPS)
        generate_kernel(_sources ${_general_template} "BinaryOp" "")
        foreach(VEC_KERNEL_NAME IN LISTS VEC_KERNELS)
          foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
            generate_kernel(_sources ${_general_template} ${VEC_KERNEL_NAME} ", ${VECTOR_WIDTH}")
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  set(${INST_BINARYOP_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_binary(
  OUTPUT_VAR binary_kernels
  FILENAME   binaryop
)

snn_object_library(
  WITH_SYCL
  TARGET 
    binaryop
  KERNEL_SOURCES
    ${binary_kernels}
  SOURCES
    launch_binaryop.cc
)
