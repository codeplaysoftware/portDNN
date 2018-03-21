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
project(snn_tests)
include(HandleGTest)
include(SNNHelpers)

snn_test(
  WITH_SYCL
  TARGET
    basic_sycl
  SOURCES
    basic_sycl_test.cc
)
snn_test(
  WITH_SYCL
  TARGET
    fast_divison
  SOURCES
    helpers/fast_div.cc
)
snn_test(
  TARGET
    common_macros
  SOURCES
    helpers/common_macros.cc
)
snn_test(
  TARGET
    non_device_macros
  SOURCES
    helpers/non_device_macros.cc
)
snn_test(
  TARGET
    device_macros
  SOURCES
    helpers/device_macros.cc
)
snn_test(
  TARGET
    min_max
  SOURCES
    helpers/min_max.cc
)
snn_test(
  TARGET
    ratio_helpers
  SOURCES
    helpers/ratio.cc
)
snn_test(
  TARGET
    input_window_index_helpers
  SOURCES
    helpers/input_window_index.cc
)
snn_test(
  TARGET
    output_window_index_helpers
  SOURCES
    helpers/output_window_index.cc
)
