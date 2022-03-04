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

macro(instantiate_depth_conv_impl out_var)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_DEPTH_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}")
  set(_filename "${_filename}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/depthwise_conv2d/${_filename})
  configure_file(${INST_DEPTH_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(instantiate_depthwise_conv)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_DEPTH
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
        instantiate_depth_conv_impl(_sources)
      endforeach()
    endforeach()
  endforeach()
  set(${INST_DEPTH_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

instantiate_depthwise_conv(
  OUTPUT_VAR    depth_conv2d_kernel_sources
  TEMPLATE_FILE queue_depthwise_conv2d.cc.in
  FILENAME      depthwise
)

snn_object_library(
  WITH_SYCL
  TARGET depthwise_conv2d
  SOURCES launch.cc
  KERNEL_SOURCES ${depth_conv2d_kernel_sources}
)

