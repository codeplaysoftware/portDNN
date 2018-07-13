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

set(SNN_CONV_TYPES conv_type::Forward
                   conv_type::InputBackprop
                   conv_type::FilterBackprop
)

macro(instantiate_direct_conv_impl out_var window stride)
  list(FIND SNN_CONV_TYPES ${CONV_TYPE} CONV_TYPE_IDX)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename
    "${INST_DIRECT_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${CONV_TYPE_IDX}_${window}_${stride}.cc"
  )
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/direct/${_filename})
  set(DIRECT_WINDOW ${window})
  set(DIRECT_STRIDE ${stride})
  configure_file(${INST_DIRECT_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(instantiate_direct_conv)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_DIRECT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(CONV_TYPE IN LISTS SNN_CONV_TYPES)
        instantiate_direct_conv_impl(_sources 0 0)
        if(SNN_CONV2D_DIRECT_STATIC_KERNELS)
          instantiate_direct_conv_impl(_sources 1 1)
          instantiate_direct_conv_impl(_sources 3 1)
          instantiate_direct_conv_impl(_sources 3 2)
          instantiate_direct_conv_impl(_sources 5 1)
          instantiate_direct_conv_impl(_sources 5 2)
        endif()
      endforeach()
    endforeach()
  endforeach()
  set(${INST_DIRECT_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

instantiate_direct_conv(
  OUTPUT_VAR    direct_conv2d_kernel_sources
  TEMPLATE_FILE conv2d/direct/direct_impl_tpl.cc.in
  FILENAME      dc2d
)
snn_object_library(
  WITH_SYCL
  TARGET direct_conv2d
  SOURCES conv2d/direct/launch_direct.cc
  KERNEL_SOURCES ${direct_conv2d_kernel_sources}
)

macro(instantiate_tiled_conv_impl out_var window stride tile_row tile_col
                                  channel_vector feature_vector)
  list(FIND SNN_CONV_TYPES ${CONV_TYPE} CONV_TYPE_IDX)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_TILED_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}")
  set(_filename "${_filename}_${CONV_TYPE_IDX}_${tile_row}_${tile_col}")
  set(_filename
    "${_filename}_${channel_vector}_${feature_vector}_${window}_${stride}.cc"
  )
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/tiled/${_filename})
  set(TILE_ROW ${tile_row})
  set(TILE_COL ${tile_col})
  set(CHANNEL_VECTOR ${channel_vector})
  set(FEATURE_VECTOR ${feature_vector})
  set(WINDOW ${window})
  set(STRIDE ${stride})
  configure_file(${INST_TILED_TEMPLATE_FILE} ${_gen_file})
  if(COMPUTECPP_FGLRX_WORKAROUND AND ${window} EQUAL 1 AND ${stride} EQUAL 1)
    # Workaround an AMD OpenCL compiler bug
    set_property(SOURCE ${_gen_file} PROPERTY COMPUTECPP_SOURCE_FLAGS "-O2")
  endif()
  list(APPEND ${out_var} ${_gen_file})
endmacro()
function(instantiate_tiled_conv)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_TILED
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(CONV_TYPE IN LISTS SNN_CONV_TYPES)
        # The following tile sizes and kernel parameters should match those
        # required in sycldnn::conv2d::launch_tiled() function defined in
        # src/conv2d/tiled/launch_tiled.cc
        if(NOT CONV_TYPE STREQUAL "conv_type::FilterBackprop")
          instantiate_tiled_conv_impl(_sources 3 1 1 4 1 1)
          instantiate_tiled_conv_impl(_sources 3 2 2 4 1 4)
          if(CONV_TYPE STREQUAL "conv_type::Forward")
            instantiate_tiled_conv_impl(_sources 1 2 1 2 1 4)
            instantiate_tiled_conv_impl(_sources 1 2 1 2 1 1)
            instantiate_tiled_conv_impl(_sources 3 2 2 2 1 4)
            instantiate_tiled_conv_impl(_sources 3 2 2 2 1 1)
          endif()
          if(CONV_TYPE STREQUAL "conv_type::InputBackprop")
            instantiate_tiled_conv_impl(_sources 1 2 2 2 1 4)
            instantiate_tiled_conv_impl(_sources 1 2 2 2 1 1)
            instantiate_tiled_conv_impl(_sources 3 2 2 4 1 2)
            instantiate_tiled_conv_impl(_sources 3 1 3 4 1 4)
          endif()
          instantiate_tiled_conv_impl(_sources 3 1 2 2 1 4)
          instantiate_tiled_conv_impl(_sources 3 1 3 4 1 1)
          instantiate_tiled_conv_impl(_sources 3 2 2 2 1 1)
          instantiate_tiled_conv_impl(_sources 5 1 2 2 1 2)
          instantiate_tiled_conv_impl(_sources 5 1 2 4 1 1)
          instantiate_tiled_conv_impl(_sources 1 1 2 2 1 4)
          instantiate_tiled_conv_impl(_sources 1 1 2 2 1 1)
          instantiate_tiled_conv_impl(_sources 1 2 2 2 1 1)
        endif()
      endforeach()
    endforeach()
  endforeach()
  set(${INST_TILED_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

instantiate_tiled_conv(
  OUTPUT_VAR    tiled_conv2d_kernel_sources
  TEMPLATE_FILE conv2d/tiled/tiled_impl_tpl.cc.in
  FILENAME      tc2d
)
snn_object_library(
  WITH_SYCL
  TARGET tiled_conv2d
  SOURCES conv2d/tiled/launch_tiled.cc
  KERNEL_SOURCES ${tiled_conv2d_kernel_sources}
)

snn_object_library(
  WITH_SYCL
  TARGET pooling
  KERNEL_SOURCES
    pooling/launch_max_pooling.cc
    pooling/launch_avg_pooling.cc
)
