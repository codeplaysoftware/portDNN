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
#
cmake_minimum_required(VERSION 3.2.2)
include(SNNHelpers)

set(SNN_CONV_TYPES conv_type::Forward
                   conv_type::InputBackprop
                   conv_type::FilterBackprop
)

macro(instantiate_direct_conv_impl window stride)
  list(FIND SNN_CONV_TYPES ${CONV_TYPE} CONV_TYPE_IDX)
  set(_filename
    "${INST_DIRECT_FILENAME}_${DATA_TYPE}_${INDEX_TYPE}_${CONV_TYPE_IDX}_${window}_${stride}.cc"
  )
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/conv2d/direct/${_filename})
  set(DIRECT_WINDOW ${window})
  set(DIRECT_STRIDE ${stride})
  configure_file(${INST_DIRECT_TEMPLATE_FILE} ${_gen_file})
  target_sources(${INST_DIRECT_TARGET_LIB} PRIVATE ${_gen_file})
endmacro()

function(instantiate_direct_conv)
  set(options)
  set(one_value_args
    TARGET_LIB
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
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(CONV_TYPE IN LISTS SNN_CONV_TYPES)
        instantiate_direct_conv_impl(0 0)
        if(SNN_CONV2D_DIRECT_STATIC_KERNELS)
          instantiate_direct_conv_impl(1 1)
          instantiate_direct_conv_impl(3 1)
          instantiate_direct_conv_impl(3 2)
          instantiate_direct_conv_impl(5 1)
          instantiate_direct_conv_impl(5 2)
        endif()
      endforeach()
    endforeach()
  endforeach()
endfunction()

add_library(direct_conv2d_kernels "")
instantiate_direct_conv(
  TARGET_LIB    direct_conv2d_kernels
  TEMPLATE_FILE conv2d/direct/direct_impl_tpl.cc.in
  FILENAME      dc2d
)
# Need to call this after setting the sources to ensure that all sources are
# compiled with SYCL support.
snn_target(
  WITH_SYCL
  TARGET direct_conv2d_kernels
)
add_library(direct_conv2d conv2d/direct/launch_direct.cc)
target_link_libraries(direct_conv2d INTERFACE direct_conv2d_kernels)
snn_target(
  WITH_SYCL
  TARGET direct_conv2d
)

