/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PORTDNN_TEST_BACKEND_BACKEND_TEST_FIXTURE_H_
#define PORTDNN_TEST_BACKEND_BACKEND_TEST_FIXTURE_H_
#include <gtest/gtest.h>

#include "src/backend/snn_backend_provider.h"
#include "src/backend/snn_usm_backend_provider.h"

#if defined(SNN_TEST_EIGEN) || defined(SNN_TEST_EIGEN_MATMULS)
#include <unsupported/Eigen/CXX11/Tensor>

#include "portdnn/backend/eigen_backend.h"

#include "src/backend/eigen_backend_provider.h"
#include "src/backend/eigen_backend_snn_matmul_provider.h"
#endif

#if defined(SNN_TEST_SYCLBLAS) || defined(SNN_TEST_SYCLBLAS_MATMULS)
#include "portdnn/backend/sycl_blas_backend.h"
#include "src/backend/syclblas_backend_provider.h"
#endif  // SNN_TEST_SYCLBLAS_MATMULS

#if defined(SNN_TEST_CLBLAST) || defined(SNN_TEST_CLBLAST_MATMULS)
#include "portdnn/backend/clblast_backend.h"
#include "src/backend/clblast_backend_provider.h"
#endif  // SNN_TEST_CLBLAST

template <typename Backend>
struct BackendTestFixture;

template <typename Backend>
struct BackendTestFixture : public ::testing::Test {
 public:
  BackendTestFixture() : provider_{} {}
  /** TearDown() method called upon termination of test fixture. */
  void TearDown() override {}

 protected:
  sycldnn::backend::BackendProvider<Backend> provider_;
};

#if defined(SNN_TEST_EIGEN) || defined(SNN_TEST_EIGEN_MATMULS)
template <>
inline void BackendTestFixture<sycldnn::backend::EigenBackend>::TearDown() {
  auto& device = provider_.get_eigen_device();
  device.sycl_queue().wait_and_throw();
  device.deallocate_all();
}

template <>
inline void
BackendTestFixture<sycldnn::backend::EigenBackendSNNMatmul>::TearDown() {
  auto& device = provider_.get_eigen_device();
  device.sycl_queue().wait_and_throw();
  device.deallocate_all();
}
#endif  // SNN_TEST_EIGEN || SNN_TEST_EIGEN_MATMULS

#if defined(SNN_TEST_SYCLBLAS) || defined(SNN_TEST_SYCLBLAS_MATMULS)
template <>
inline void BackendTestFixture<sycldnn::backend::SyclBLASBackend>::TearDown() {
  provider_.get_default_queue().wait_and_throw();
}
#endif  // SNN_TEST_SYCLBLAS || SNN_TEST_SYCLBLAS_MATMULS

#if defined(SNN_TEST_CLBLAST) || defined(SNN_TEST_CLBLAST_MATMULS)
template <>
inline void BackendTestFixture<sycldnn::backend::CLBlastBackend>::TearDown() {
  provider_.get_backend().get_queue().wait_and_throw();
}
#endif  // SNN_TEST_CLBLAST || SNN_TEST_CLBLAST_MATMULS

#endif  // PORTDNN_TEST_BACKEND_BACKEND_TEST_FIXTURE_H_
