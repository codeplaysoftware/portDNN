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
cmake_minimum_required(VERSION 3.2.2)

include(HandleGTest)
include(SNNHelpers)

foreach(_gather_inp_dim IN ITEMS 1 2 3)
  snn_test(
    WITH_SYCL
    TARGET
      gather_${_gather_inp_dim}d
    SIZE
      moderate
    SOURCES
      gather_${_gather_inp_dim}d.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endforeach()

snn_test(
  WITH_SYCL
  TARGET
    gather_cast
  SOURCES
    gather_cast.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)

snn_test(
  WITH_SYCL
  TARGET
    gather_indices
  SOURCES
    gather_indices.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)

if(SNN_ENABLE_USM)
  snn_test(
    WITH_SYCL
    TARGET
      gather_event_dependencies
    SOURCES
      gather_event_dependencies.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endif()
