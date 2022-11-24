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
    softmax_test
  SOURCES
    softmax_forward.cc
    softmax_grad.cc
  PUBLIC_LIBRARIES
    sycl_dnn
)

if(SNN_ENABLE_USM)
  snn_test(
    WITH_SYCL
    TARGET
      softmax_event_dependencies_forward_test
    SOURCES
      softmax_event_dependencies_forward.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
  snn_test(
    WITH_SYCL
    TARGET
      softmax_event_dependencies_grad_test
    SOURCES
      softmax_event_dependencies_grad.cc
    PUBLIC_LIBRARIES
      sycl_dnn
  )
endif()
