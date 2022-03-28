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

if(SNN_TEST_EIGEN OR SNN_TEST_SYCLBLAS)
  set(_cxx_opts CXX_OPTS)
  set(_matmul_backends)
  set(_matmul_providers)

  if(SNN_TEST_EIGEN)
    include(HandleEigen)
    list(APPEND _cxx_opts -DSNN_TEST_EIGEN=1)
    list(APPEND _matmul_backends EigenBackendSNNMatmul EigenBackend)
    list(APPEND _matmul_providers Eigen::Eigen)
  endif()

  if(SNN_TEST_SYCLBLAS)
    include(Handlesycl_blas)
    list(APPEND _cxx_opts -DSNN_TEST_SYCLBLAS=1)
    list(APPEND _matmul_backends SyclBLASBackend)
    list(APPEND _matmul_providers SYCL_BLAS::sycl_blas)
  endif()
  if(SNN_TEST_CLBLAST)
    include(HandleCLBlast)
    list(APPEND _cxx_opts -DSNN_TEST_CLBLAST=1)
    list(APPEND _matmul_backends CLBlastBackend)
    list(APPEND _matmul_providers clblast)
  endif()

  add_library(backend_matmul "")
  foreach(BACKEND_TYPE IN LISTS _matmul_backends)
    foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
      # The Eigen tests only require int32, not int64 so to save time only
      # compile for 32 bits.
      foreach(INDEX_TYPE IN ITEMS int32_t)
        string(MAKE_C_IDENTIFIER ${BACKEND_TYPE} BACKEND_ID)
        string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
        set(_filename matmul_${BACKEND_ID}_${DTYPE_ID}_${INDEX_TYPE}.cc)
        set(_gen_dir ${CMAKE_BINARY_DIR}/generated/test/backend)
        configure_file(matmul_tpl.cc.in ${_gen_dir}/${_filename})
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
      sycl_dnn
      ${_matmul_providers}
      GTest::GTest
    ${_cxx_opts}
  )
  snn_test(
    WITH_SYCL
    TARGET
      matmul_backend
    SIZE
      long
    SOURCES
      matmul_backend.cc
    PUBLIC_LIBRARIES
      ${_matmul_providers}
      backend_matmul
    ${_cxx_opts}
  )
  if(SNN_TEST_EIGEN)
    snn_test(
      WITH_SYCL
      TARGET
        backend_death_tests
      SOURCES
        backend_death_tests.cc
      PUBLIC_LIBRARIES
        ${_matmul_providers}
      ${_cxx_opts}
    )
    snn_test(
      WITH_SYCL
      TARGET
        eigen_external_handler
      KERNEL_SOURCES
        eigen_external_handler.cc
      PUBLIC_LIBRARIES
        ${_matmul_providers}
      ${_cxx_opts}
    )
    snn_test(
      WITH_SYCL
      TARGET
        eigen_internal_handler
      KERNEL_SOURCES
        eigen_internal_handler.cc
      PUBLIC_LIBRARIES
        ${_matmul_providers}
      ${_cxx_opts}
    )
  endif()
endif()
