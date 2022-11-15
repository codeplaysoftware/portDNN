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

foreach(_type IN ITEMS "relu" "tanh" "log" "floor" "sqrt")
  foreach(_direction IN ITEMS "forward" "grad")
    if (NOT _type STREQUAL "floor" OR NOT _direction STREQUAL "grad")
      snn_test(
        WITH_SYCL
        TARGET
          pointwise_${_type}_${_direction}
        SIZE
          short
        SOURCES
          ${_type}_${_direction}.cc
        PUBLIC_LIBRARIES
          sycl_dnn
      )
    endif()
  endforeach()
endforeach()

if(SNN_ENABLE_USM)
  snn_test(
    WITH_SYCL
    TARGET
      pointwise_event_dependencies
    SIZE
      short
    SOURCES
      pointwise_event_dependencies.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endif()