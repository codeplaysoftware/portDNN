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

foreach(_trans_lhs IN ITEMS "true" "false")
  foreach(_trans_rhs IN ITEMS "true" "false")
    foreach(_batch IN ITEMS 1 3)
      foreach(_beta IN ITEMS 0 1)
        snn_test(
          WITH_SYCL
          TARGET
            matmul_batch${_batch}_beta${_beta}_${_trans_lhs}_${_trans_rhs}
          SIZE
            moderate
          SOURCES
            matmul_batch${_batch}_beta${_beta}_${_trans_lhs}_${_trans_rhs}.cc
          PUBLIC_LIBRARIES
            sycl_dnn
        )
      endforeach()
    endforeach()
  endforeach()
endforeach()

snn_test(
  WITH_SYCL
  TARGET
    matmul_offset
  SIZE
    moderate
  SOURCES
    matmul_offset.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)


if(SNN_ENABLE_USM)
  snn_test(
    WITH_SYCL
    TARGET
      matmul_event
    SOURCES
      matmul_event_dependency.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endif()