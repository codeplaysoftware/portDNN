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

foreach(_transpose_dim IN ITEMS 2 3 4)
  snn_test(
    WITH_SYCL
    TARGET
      transpose_${_transpose_dim}d
    SIZE
      moderate
    SOURCES
      transpose_${_transpose_dim}d.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endforeach()

snn_test(
  WITH_SYCL
  TARGET
    transpose_conversions
  SOURCES
    conversions.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)
snn_test(
  WITH_SYCL
  TARGET
    transpose_offsets
  SOURCES
    offsets.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)
snn_test(
  WITH_SYCL
  TARGET
    transpose_cast
  SOURCES
    transpose_cast.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)

if(SNN_ENABLE_USM)
  snn_test(
    WITH_SYCL
    TARGET
      transpose_event
    SOURCES
      transpose_event_dependencies.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endif()
