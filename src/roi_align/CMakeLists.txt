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

macro(generate_kernel out_var template op)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_ROI_ALIGN_FILENAME}_${DTYPE_ID}_${BI_TYPE}_${INDEX_TYPE}_${op}")
  set(_filename "${_filename}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/roi_align/${_filename})
  set(OPERATOR ${op})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_roi_align)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_ROI_ALIGN
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_general_template queue_roi_align_kernel_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(BI_TYPE IN ITEMS int32_t int64_t) 
        generate_kernel(_sources ${_general_template} AveragePool)
        generate_kernel(_sources ${_general_template} MaxPool)
      endforeach()
    endforeach()
  endforeach()
  set(${INST_ROI_ALIGN_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_roi_align(
  OUTPUT_VAR roi_align_kernels
  FILENAME   roi_align
)

snn_object_library(
  WITH_SYCL
  TARGET roi_align
  KERNEL_SOURCES
    ${roi_align_kernels}
  SOURCES
    launch_roi_align.cc
)
