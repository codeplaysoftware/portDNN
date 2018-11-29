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

include(HandleGTest)
include(SNNHelpers)

if(SNN_TEST_EIGEN)
  # windows/strides are different to conv2d here - Tensorflow doesn't allow
  # stride > window
  set(pooling_windows 1 3 3 5 5 7 7 11 11)
  set(pooling_strides 1 1 2 1 2 1 4 1 4)
  list(LENGTH pooling_windows pooling_list_len_raw)
  math(EXPR pooling_list_len "${pooling_list_len_raw} - 1")
  foreach(_index RANGE ${pooling_list_len})
    list(GET pooling_windows ${_index} _window)
    list(GET pooling_strides ${_index} _stride)
    foreach(_type IN ITEMS "max" "avg")
      foreach(_direction IN ITEMS "forward" "grad")
        if((_type STREQUAL "avg") AND
          (_window EQUAL 11 OR ((_window EQUAL 7) AND (_stride EQUAL 4))))
          message(STATUS "Skipping avg_window${_window}_stride${_stride}_${_direction}")
        else()
          snn_test(
            WITH_SYCL
            TARGET
              pooling_${_type}_window${_window}_stride${_stride}_${_direction}
            SIZE
              moderate
            SOURCES
              ${_type}_window${_window}_stride${_stride}_${_direction}.cc
            OBJECTS
              $<TARGET_OBJECTS:pooling>
            PUBLIC_LIBRARIES
              Eigen::Eigen
          )
        endif()
      endforeach()
    endforeach()
  endforeach()
endif(SNN_TEST_EIGEN)
