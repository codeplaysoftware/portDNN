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

foreach(_dir IN ITEMS forward gradient)
  foreach(_op IN ITEMS training frozen)
    set(_target batchnorm_${_dir}_${_op})
    snn_test(
      WITH_SYCL
      TARGET
        ${_target}
      SIZE
        moderate
      SOURCES
        ${_target}.cc
      PUBLIC_LIBRARIES
        sycl_dnn
    )
    if (SNN_ENABLE_USM)
      set(_target batchnorm_event_dependencies_${_dir}_${_op})
      snn_test(
        WITH_SYCL
        TARGET
          ${_target}
        SIZE
          moderate
        SOURCES
          ${_target}.cc
        PUBLIC_LIBRARIES
          sycl_dnn
      )
    endif()
  endforeach()
endforeach()
