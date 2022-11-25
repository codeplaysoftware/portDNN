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

include(HandleGTest)
include(SNNHelpers)

snn_test(
  WITH_SYCL
  TARGET
    simple_depthwise_conv2d
  SIZE
    moderate
  SOURCES
    simple_depthwise.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)

foreach(_type IN ITEMS "forward" "input_backprop" "filter_backprop")
  snn_test(
    WITH_SYCL
    TARGET
      depthwise_conv2d_${_type}_1x1
    SIZE
      short
    SOURCES
      ${_type}_1x1.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endforeach()

set(_windows 1 1 3 3 5 5 7 7 11 11)
set(_strides 1 2 1 2 1 2 1 4 1 4)
list(LENGTH _windows _list_len_raw)
math(EXPR _list_len "${_list_len_raw} - 1")
foreach(_index RANGE ${_list_len})
  list(GET _windows ${_index} _window)
  list(GET _strides ${_index} _stride)
  set(_test_size moderate)
  if (_window GREATER_EQUAL 11)
    set(_test_size long)
  endif()
  foreach(_type IN ITEMS "forward" "input_backprop" "filter_backprop")
    snn_test(
      WITH_SYCL
      TARGET
        depthwise_conv2d_${_type}_window${_window}_stride${_stride}
      SIZE
        ${_test_size}
      SOURCES
        ${_type}_window${_window}_stride${_stride}.cc
      PUBLIC_LIBRARIES
        sycl_dnn
    )
  endforeach()
endforeach()

if(SNN_ENABLE_USM)
  snn_test(
    WITH_SYCL
    TARGET
      depthwise_conv2d_event_dependencies
    SIZE
      moderate
    SOURCES
      depthwise_conv2d_event_dependencies.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endif()
