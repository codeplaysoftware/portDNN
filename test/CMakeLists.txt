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
snn_test(
  WITH_SYCL
  TARGET
    tensor_index
  SOURCES
    helpers/tensor_index.cc
)
snn_test(
  TARGET
    round_to_power_two
  SOURCES
    helpers/round_power_two.cc
)
snn_test(
  WITH_SYCL
  TARGET
    math_helpers
  SOURCES
    helpers/math.cc
)
snn_test(
  WITH_SYCL
  TARGET
    vector_element
  SOURCES
    helpers/vector_element.cc
)
snn_test(
  TARGET
    padding_helpers
  SOURCES
    helpers/padding.cc
)
snn_test(
  TARGET
    add_padding_to_conv2dparams
  SOURCES
    conv2d/helpers/add_padding_to_params.cc
)
if(SNN_TEST_EIGEN)
  add_library(eigen_matmul "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    # The Eigen tests only require int32, not int64 so to save time only
    # compile for 32 bits.
    foreach(INDEX_TYPE IN ITEMS int32_t)
      set(_filename eigen_matmul_${DATA_TYPE}_${INDEX_TYPE}.cc)
      set(_gen_dir ${CMAKE_BINARY_DIR}/generated/test/backend)
      configure_file(backend/eigen_matmul_tpl.cc.in ${_gen_dir}/${_filename})
      target_sources(eigen_matmul PRIVATE ${_gen_dir}/${_filename})
    endforeach()
  endforeach()
  # Need to call this after setting the sources to ensure that all sources are
  # compiled with SYCL support.
  get_target_property(eigen_matmul_sources eigen_matmul SOURCES)
  snn_target(
    TARGET eigen_matmul
    SOURCES ${eigen_matmul_sources}
    PUBLIC_LIBRARIES
      Eigen::Eigen
      GTest::GTest
  )
  snn_test(
    WITH_SYCL
    TARGET
      eigen_death_tests
    SOURCES
      backend/eigen_backend_death_tests.cc
    PUBLIC_LIBRARIES
      Eigen::Eigen
  )
  snn_test(
    WITH_SYCL
    TARGET
      eigen_external_handler
    SOURCES
      backend/eigen_external_handler.cc
    PUBLIC_LIBRARIES
      Eigen::Eigen
  )
  snn_test(
    WITH_SYCL
    TARGET
      eigen_internal_handler
    SOURCES
      backend/eigen_internal_handler.cc
    PUBLIC_LIBRARIES
      Eigen::Eigen
  )
  snn_test(
    WITH_SYCL
    TARGET
      eigen_matmul_backend
    SIZE
      long
    SOURCES
      backend/eigen_matmul_backend.cc
    PUBLIC_LIBRARIES
      Eigen::Eigen
      eigen_matmul
  )
  snn_test(
    WITH_SYCL
    TARGET
      simple_convolution
    SOURCES
      convolutions.cc
    OBJECTS
      $<TARGET_OBJECTS:direct_conv2d>
    PUBLIC_LIBRARIES
      Eigen::Eigen
  )
endif()
