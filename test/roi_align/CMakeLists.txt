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

snn_test(
  WITH_SYCL
  TARGET
    roi_align
  SOURCES
    roi_align_tests.cc
  OBJECTS
    $<TARGET_OBJECTS:roi_align>
)

if(SNN_ENABLE_USM)
snn_test(
  WITH_SYCL
  TARGET
    roi_align_event_deps
  SOURCES
    roi_align_event_dependencies_test.cc
  OBJECTS
    $<TARGET_OBJECTS:roi_align>
)
endif()
