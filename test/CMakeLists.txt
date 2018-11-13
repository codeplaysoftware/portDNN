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
  KERNEL_SOURCES
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
    add_padding_to_params
  SOURCES
    helpers/add_padding_to_params.cc
)
snn_test(
  WITH_SYCL
  TARGET
    selector
  SOURCES
    conv2d/default_selector.cc
  OBJECTS
    $<TARGET_OBJECTS:direct_conv2d>
    $<TARGET_OBJECTS:selector_conv2d>
)
if(SNN_TEST_EIGEN)
  list(APPEND _matmul_backends EigenBackendSNNMatmul)
  list(APPEND _matmul_providers Eigen::Eigen)

  list(APPEND _cxx_opts CXX_OPTS)
  if(SNN_TEST_EIGEN_MATMULS)
    list(APPEND _cxx_opts -DSNN_TEST_EIGEN_MATMULS=1)
    list(APPEND _matmul_backends EigenBackend)
  endif()

  if(SNN_TEST_SYCLBLAS_MATMULS)
    list(APPEND _cxx_opts -DSNN_TEST_SYCLBLAS_MATMULS=1)
    list(APPEND _matmul_backends SyclBLASBackend)
    list(APPEND _matmul_providers SyclBLAS::SyclBLAS)
  endif()

  add_library(backend_matmul "")
  foreach(BACKEND_TYPE IN LISTS _matmul_backends)
    foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
      # The Eigen tests only require int32, not int64 so to save time only
      # compile for 32 bits.
      foreach(INDEX_TYPE IN ITEMS int32_t)
        string(MAKE_C_IDENTIFIER ${BACKEND_TYPE} BACKEND_ID)
        string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
        set(_filename matmul_${BACKEND_TYPE}_${DTYPE_ID}_${INDEX_TYPE}.cc)
        set(_gen_dir ${CMAKE_BINARY_DIR}/generated/test/backend)
        configure_file(backend/matmul_tpl.cc.in ${_gen_dir}/${_filename})
        target_sources(backend_matmul PRIVATE ${_gen_dir}/${_filename})
      endforeach()
    endforeach()
  endforeach()
  get_target_property(backend_matmul_sources backend_matmul SOURCES)
  snn_target(
    WITH_SYCL
    TARGET backend_matmul
    KERNEL_SOURCES ${backend_matmul_sources}
    PUBLIC_LIBRARIES
      ${_matmul_providers}
      GTest::GTest
    ${_cxx_opts}
  )
  snn_test(
    WITH_SYCL
    TARGET
      backend_death_tests
    SOURCES
      backend/backend_death_tests.cc
    PUBLIC_LIBRARIES
      ${_matmul_providers}
  )
  snn_test(
    WITH_SYCL
    TARGET
      eigen_external_handler
    KERNEL_SOURCES
      backend/eigen_external_handler.cc
    PUBLIC_LIBRARIES
      ${_matmul_providers}
  )
  snn_test(
    WITH_SYCL
    TARGET
      eigen_internal_handler
    KERNEL_SOURCES
      backend/eigen_internal_handler.cc
    PUBLIC_LIBRARIES
      ${_matmul_providers}
  )
  snn_test(
    WITH_SYCL
    TARGET
      matmul_backend
    SIZE
      long
    SOURCES
      backend/matmul_backend.cc
    OBJECTS
      $<TARGET_OBJECTS:matmul>
    PUBLIC_LIBRARIES
      ${_matmul_providers}
      backend_matmul
    ${_cxx_opts}
  )
endif()

add_subdirectory(matmul)
add_subdirectory(conv2d)
add_subdirectory(depthwise_conv2d)
add_subdirectory(pointwise)
add_subdirectory(pooling)
