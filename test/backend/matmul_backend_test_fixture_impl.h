/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_IMPL_H_
#define SYCLDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_IMPL_H_
#include "test/backend/matmul_backend_test_fixture.h"

template <typename Backend>
template <typename T>
void BackendMatmul<Backend>::copy_to_device(T* src, T* dst, size_t nelems) {
  auto& device = this->backend_.get_eigen_device();
  device.memcpyHostToDevice(dst, src, nelems * sizeof(T));
}

template <typename Backend>
template <typename T>
void BackendMatmul<Backend>::copy_to_host(T* src, T* dst, size_t nelems) {
  auto& device = this->backend_.get_eigen_device();
  device.memcpyDeviceToHost(dst, src, nelems * sizeof(T));
}

#ifdef SNN_TEST_SYCLBLAS_MATMULS
template <>
template <typename T>
void BackendMatmul<sycldnn::backend::SyclBLASBackend>::copy_to_device(
    T* src, T* dst, size_t nelems) {
  auto complete =
      this->backend_.get_executor().copy_to_device(src, dst, nelems);
  complete.wait_and_throw();
}

template <>
template <typename T>
void BackendMatmul<sycldnn::backend::SyclBLASBackend>::copy_to_host(
    T* src, T* dst, size_t nelems) {
  auto complete = this->backend_.get_executor().copy_to_host(src, dst, nelems);
  complete.wait_and_throw();
}
#endif  // SNN_TEST_SYCLBLAS_MATMULS

template <typename Backend>
template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
void BackendMatmul<Backend>::test_square_matmul(std::vector<T>& lhs,
                                                std::vector<T>& rhs,
                                                std::vector<T>& expected,
                                                Index dim) {
  test_nonsquare_matmul<TransposeLHS, TransposeRHS>(lhs, rhs, expected, dim,
                                                    dim, dim);
}

template <typename Backend>
template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
void BackendMatmul<Backend>::test_nonsquare_matmul(std::vector<T>& lhs,
                                                   std::vector<T>& rhs,
                                                   std::vector<T>& expected,
                                                   Index m, Index n, Index k) {
  const auto lhs_size = m * k * sizeof(T);
  const auto rhs_size = k * n * sizeof(T);
  const auto out_size = m * n * sizeof(T);
  T* lhs_ptr = this->backend_.template allocate<T>(lhs_size);
  T* rhs_ptr = this->backend_.template allocate<T>(rhs_size);
  T* out_ptr = this->backend_.template allocate<T>(out_size);

  this->copy_to_device(lhs.data(), lhs_ptr, m * k);
  this->copy_to_device(rhs.data(), rhs_ptr, k * n);

  this->backend_.template matmul<TransposeLHS, TransposeRHS>(
      lhs_ptr, rhs_ptr, out_ptr, static_cast<T>(0), m, k, n);

  std::vector<T> out(m * n);
  this->copy_to_host(out_ptr, out.data(), m * n);

  for (int i = 0; i < m * n; ++i) {
    EXPECT_EQ(expected[i], out[i]);
  }
}

template <typename Backend>
template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
void BackendMatmul<Backend>::test_square_batch_matmul(std::vector<T>& lhs,
                                                      std::vector<T>& rhs,
                                                      std::vector<T>& expected,
                                                      Index batch, Index dim) {
  const auto size = batch * dim * dim * sizeof(T);
  T* lhs_ptr = this->backend_.template allocate<T>(size);
  T* rhs_ptr = this->backend_.template allocate<T>(size);
  T* out_ptr = this->backend_.template allocate<T>(size);

  this->copy_to_device(lhs.data(), lhs_ptr, batch * dim * dim);
  this->copy_to_device(rhs.data(), rhs_ptr, batch * dim * dim);

  this->backend_.template batch_matmul<TransposeLHS, TransposeRHS>(
      lhs_ptr, rhs_ptr, out_ptr, batch, dim, dim, dim);

  std::vector<T> out(batch * dim * dim);
  this->copy_to_host(out_ptr, out.data(), batch * dim * dim);
  for (int i = 0; i < batch * dim * dim; ++i) {
    EXPECT_EQ(expected[i], out[i]);
  }
}
#endif  // SYCLDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_IMPL_H_
