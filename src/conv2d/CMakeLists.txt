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

set(SNN_CONV_TYPES conv_type::Forward
                   conv_type::InputBackprop
                   conv_type::FilterBackprop
)

macro(instantiate_direct_conv_impl out_var window stride)
  list(FIND SNN_CONV_TYPES ${CONV_TYPE} CONV_TYPE_IDX)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_DIRECT_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${CONV_TYPE_IDX}")
  set(_filename "${_filename}_${window}_${stride}_${VECTOR_WIDTH}_${LAYOUT}.cc")
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
        foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
          foreach(LAYOUT IN LISTS SNN_LAYOUTS)
            # NCHW only supports VectorWidth of 1.
            if (LAYOUT STREQUAL "NCHW" AND NOT VECTOR_WIDTH EQUAL 1)
              continue()
            endif()
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
    endforeach()
  endforeach()
  set(${INST_DIRECT_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

instantiate_direct_conv(
  OUTPUT_VAR    direct_conv2d_kernel_sources
  TEMPLATE_FILE direct/direct_impl_tpl.cc.in
  FILENAME      dc2d
)
snn_object_library(
  WITH_SYCL
  TARGET direct_conv2d
  SOURCES direct/launch_direct.cc
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
        # required in sycldnn::conv2d::launch_tiled_impl() function defined in
        # src/conv2d/tiled/launch_tiled.cc
        # TODO(dmcbain): Some of these are commented out as they are
        # duplicates of earlier list entries.
        if(NOT CONV_TYPE STREQUAL "conv_type::FilterBackprop")
          if(CONV_TYPE STREQUAL "conv_type::Forward")
            # PowerVR
            instantiate_tiled_conv_impl(_sources 3 1 5 4 2 4)
            instantiate_tiled_conv_impl(_sources 3 1 3 4 1 8)
            instantiate_tiled_conv_impl(_sources 3 1 4 3 8 2)
            instantiate_tiled_conv_impl(_sources 3 1 5 4 1 4)
            instantiate_tiled_conv_impl(_sources 3 1 5 4 8 1)
            instantiate_tiled_conv_impl(_sources 3 1 5 5 1 1)
            instantiate_tiled_conv_impl(_sources 1 1 5 2 8 8)
            instantiate_tiled_conv_impl(_sources 1 1 4 4 1 8)
            instantiate_tiled_conv_impl(_sources 1 1 5 5 8 1)
            instantiate_tiled_conv_impl(_sources 1 1 5 4 1 1)
            # ARM GPU
            instantiate_tiled_conv_impl(_sources 3 1 5 4 1 1)
            instantiate_tiled_conv_impl(_sources 1 1 2 4 4 2)
            instantiate_tiled_conv_impl(_sources 1 1 2 3 1 4)
            instantiate_tiled_conv_impl(_sources 1 1 3 4 4 1)
            instantiate_tiled_conv_impl(_sources 1 1 2 4 1 1)
            # AMD GPU
            instantiate_tiled_conv_impl(_sources 3 1 4 5 4 2)
            instantiate_tiled_conv_impl(_sources 3 1 4 5 2 2)
            instantiate_tiled_conv_impl(_sources 3 1 5 5 4 1)
            instantiate_tiled_conv_impl(_sources 3 1 4 3 1 4)
            #instantiate_tiled_conv_impl(_sources 3 1 5 4 1 1)
            instantiate_tiled_conv_impl(_sources 1 1 1 5 4 8)
            instantiate_tiled_conv_impl(_sources 1 1 2 3 4 4)
            instantiate_tiled_conv_impl(_sources 1 1 2 5 1 8)
            instantiate_tiled_conv_impl(_sources 1 1 3 4 8 1)
            instantiate_tiled_conv_impl(_sources 1 1 2 3 1 1)
            # Intel GPU
            instantiate_tiled_conv_impl(_sources 3 1 3 3 1 4)
            #instantiate_tiled_conv_impl(_sources 3 1 5 4 1 1)
            instantiate_tiled_conv_impl(_sources 1 1 4 2 4 8)
            instantiate_tiled_conv_impl(_sources 1 1 3 4 1 8)
            instantiate_tiled_conv_impl(_sources 1 1 3 4 1 1)
            #Intel CPU
            instantiate_tiled_conv_impl(_sources 3 1 5 4 1 16)
            instantiate_tiled_conv_impl(_sources 3 1 4 4 1 8)
            instantiate_tiled_conv_impl(_sources 3 1 4 5 1 1)
            instantiate_tiled_conv_impl(_sources 1 1 1 4 1 16)
            instantiate_tiled_conv_impl(_sources 1 1 1 4 1 8)
            instantiate_tiled_conv_impl(_sources 1 1 1 4 1 1)

            # Generic device
            instantiate_tiled_conv_impl(_sources 1 2 1 2 1 4)
            instantiate_tiled_conv_impl(_sources 1 2 1 2 1 1)
            instantiate_tiled_conv_impl(_sources 3 2 2 2 1 4)
            #instantiate_tiled_conv_impl(_sources 3 2 2 2 1 1)
          endif()

          if(CONV_TYPE STREQUAL "conv_type::InputBackprop")
            instantiate_tiled_conv_impl(_sources 1 2 2 2 1 4)
            #instantiate_tiled_conv_impl(_sources 1 2 2 2 1 1)
            instantiate_tiled_conv_impl(_sources 3 2 2 4 1 2)
            instantiate_tiled_conv_impl(_sources 3 1 3 4 1 4)
          endif()

          # Common tiles for Forward and InputBackprop
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
  TEMPLATE_FILE tiled/tiled_impl_tpl.cc.in
  FILENAME      tc2d
)
snn_object_library(
  WITH_SYCL
  TARGET tiled_conv2d
  SOURCES tiled/launch_tiled.cc
  KERNEL_SOURCES ${tiled_conv2d_kernel_sources}
)

macro(instantiate_im2col_zero_transform_impl out_var vector)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename
    "${INST_IM2COL_ZERO_FILENAME}_${DTYPE_ID}_${vector}.cc"
  )
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/im2col/${_filename})
  set(VECTOR_WIDTH ${vector})
  configure_file(${INST_IM2COL_ZERO_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()
function(instantiate_im2col_zero_transform)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_IM2COL_ZERO
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(INST_IM2COL_ZERO)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    instantiate_im2col_zero_transform_impl(_sources 1)
    instantiate_im2col_zero_transform_impl(_sources 2)
    instantiate_im2col_zero_transform_impl(_sources 4)
  endforeach()
  set(${INST_IM2COL_ZERO_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

macro(instantiate_im2col_input_transform_impl out_var vector)
  list(FIND SNN_CONV_TYPES ${CONV_TYPE} CONV_TYPE_IDX)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_IM2COL_INPUT_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}")
  set(_filename "${_filename}_${CONV_TYPE_IDX}_${vector}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/im2col/${_filename})
  set(VECTOR_WIDTH ${vector})
  configure_file(${INST_IM2COL_INPUT_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()
function(instantiate_im2col_input_transform)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_IM2COL_INPUT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(INST_IM2COL_INPUT)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(CONV_TYPE IN LISTS SNN_CONV_TYPES)
        instantiate_im2col_input_transform_impl(_sources 1)
        instantiate_im2col_input_transform_impl(_sources 2)
        instantiate_im2col_input_transform_impl(_sources 4)
      endforeach()
    endforeach()
  endforeach()
  set(${INST_IM2COL_INPUT_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

macro(instantiate_im2col_filter_transform_impl out_var)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(_filename "${INST_IM2COL_FILTER_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/im2col/${_filename})
  configure_file(${INST_IM2COL_FILTER_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()
function(instantiate_im2col_filter_transform)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_IM2COL_FILTER
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(INST_IM2COL_FILTER)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      instantiate_im2col_filter_transform_impl(_sources)
    endforeach()
  endforeach()
  set(${INST_IM2COL_FILTER_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

instantiate_im2col_zero_transform(
  OUTPUT_VAR    im2col_zero_kernels
  TEMPLATE_FILE im2col/zero_out_impl_tpl.cc.in
  FILENAME      im2col_zero
)
instantiate_im2col_input_transform(
  OUTPUT_VAR    im2col_input_kernels
  TEMPLATE_FILE im2col/input_transform_impl_tpl.cc.in
  FILENAME      im2col_input
)
instantiate_im2col_filter_transform(
  OUTPUT_VAR    im2col_filter_kernels
  TEMPLATE_FILE im2col/filter_transform_impl_tpl.cc.in
  FILENAME      im2col_filter
)
snn_object_library(
  WITH_SYCL
  TARGET im2col_conv2d
  SOURCES im2col/launch_input_transform.cc
          im2col/launch_filter_transform.cc
  KERNEL_SOURCES ${im2col_zero_kernels}
                 ${im2col_input_kernels}
                 ${im2col_filter_kernels}
)

macro(winograd_filter_impl out_var)
  set(_filename "${WG_FILTER_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${CONV_TYPE_IDX}")
  set(_filename "${_filename}_${WINOGRAD_M}_${WINOGRAD_N}")
  set(_filename "${_filename}_${WINOGRAD_R}_${WINOGRAD_S}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/winograd/${_filename})
  configure_file(${WG_FILTER_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()

macro(winograd_input_impl out_var)
  set(_base_filename "${WG_INPUT_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${CONV_TYPE_IDX}")
  set(_base_filename "${_base_filename}_${WINOGRAD_M}_${WINOGRAD_N}")
  set(_base_filename "${_base_filename}_${WINOGRAD_R}_${WINOGRAD_S}")
  foreach(CHANNEL_VECTOR IN ITEMS 1 2 4)
    set(_filename "${_base_filename}_${CHANNEL_VECTOR}.cc")
    set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/winograd/${_filename})
    configure_file(${WG_INPUT_TEMPLATE_FILE} ${_gen_file})
    list(APPEND ${out_var} ${_gen_file})
  endforeach()
endmacro()

macro(winograd_output_impl out_var)
  set(_base_filename "${WG_OUTPUT_FILENAME}_${DTYPE_ID}_${INDEX_TYPE}_${CONV_TYPE_IDX}")
  set(_base_filename "${_base_filename}_${WINOGRAD_M}_${WINOGRAD_N}")
  set(_base_filename "${_base_filename}_${WINOGRAD_R}_${WINOGRAD_S}")
  if(CONV_TYPE STREQUAL "conv_type::FilterBackprop")
    set(_acc_list true false)
  else()
    set(_acc_list false)
  endif()
  foreach(ACCUMULATE IN LISTS _acc_list)
    set(_filename "${_base_filename}_${ACCUMULATE}.cc")
    set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/winograd/${_filename})
    configure_file(${WG_OUTPUT_TEMPLATE_FILE} ${_gen_file})
    list(APPEND ${out_var} ${_gen_file})
  endforeach()
endmacro()

macro(instantiate_winograd_impl out_var m n r s)
  list(FIND SNN_CONV_TYPES ${CONV_TYPE} CONV_TYPE_IDX)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  set(WINOGRAD_M ${m})
  set(WINOGRAD_N ${n})
  set(WINOGRAD_R ${r})
  set(WINOGRAD_S ${s})
  winograd_filter_impl(${out_var})
  winograd_input_impl(${out_var})
  winograd_output_impl(${out_var})
endmacro()

function(instantiate_winograd)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILTER_TEMPLATE_FILE
    FILTER_FILENAME
    INPUT_TEMPLATE_FILE
    INPUT_FILENAME
    OUTPUT_TEMPLATE_FILE
    OUTPUT_FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(WG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(WG)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(CONV_TYPE IN LISTS SNN_CONV_TYPES)
        instantiate_winograd_impl(_sources 3 3 3 3)
        if(CONV_TYPE STREQUAL "conv_type::FilterBackprop")
          instantiate_winograd_impl(_sources 3 3 2 2)
          instantiate_winograd_impl(_sources 3 1 2 1)
          instantiate_winograd_impl(_sources 1 3 1 2)
        else()
          instantiate_winograd_impl(_sources 4 4 3 3)
          instantiate_winograd_impl(_sources 2 2 3 3)
          instantiate_winograd_impl(_sources 2 1 3 1)
          instantiate_winograd_impl(_sources 1 2 1 3)
        endif()
      endforeach()
    endforeach()
  endforeach()
  set(${WG_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

instantiate_winograd(
  OUTPUT_VAR winograd_kernel_sources
  FILTER_TEMPLATE_FILE winograd/queue_filter_transform.cc.in
  FILTER_FILENAME winograd_filter
  INPUT_TEMPLATE_FILE winograd/queue_input_transform.cc.in
  INPUT_FILENAME winograd_input
  OUTPUT_TEMPLATE_FILE winograd/queue_output_transform.cc.in
  OUTPUT_FILENAME winograd_output
)

snn_object_library(
  WITH_SYCL
  TARGET winograd_conv2d
  KERNEL_SOURCES ${winograd_kernel_sources}
  SOURCES winograd/launch_filter_transform.cc
          winograd/launch_input_transform.cc
          winograd/launch_output_transform.cc
)

snn_object_library(
  WITH_SYCL
  TARGET selector_conv2d
  SOURCES
    selector/default_selector.cc
)
