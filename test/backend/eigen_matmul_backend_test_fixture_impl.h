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
#ifndef SYCLDNN_TEST_BACKEND_EIGEN_MATMUL_BACKEND_TEST_FIXTURE_IMPL_H_
#define SYCLDNN_TEST_BACKEND_EIGEN_MATMUL_BACKEND_TEST_FIXTURE_IMPL_H_
#include "test/backend/eigen_matmul_backend_test_fixture.h"
template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
void EigenBackendMatmul::test_square_matmul(std::vector<T>& lhs,
                                            std::vector<T>& rhs,
                                            std::vector<T>& expected,
                                            Index dim) {
  auto device = get_eigen_device();
  const auto size = dim * dim * sizeof(T);
  T* lhs_ptr = backend_.allocate<T>(size);
  T* rhs_ptr = backend_.allocate<T>(size);
  T* out_ptr = backend_.allocate<T>(size);

  device.memcpyHostToDevice(lhs_ptr, lhs.data(), size);
  device.memcpyHostToDevice(rhs_ptr, rhs.data(), size);

  backend_.matmul<TransposeLHS, TransposeRHS>(lhs_ptr, rhs_ptr, out_ptr,
                                              static_cast<T>(0), dim, dim, dim);
  std::vector<T> out(dim * dim);
  device.memcpyDeviceToHost(out.data(), out_ptr, size);
  for (int i = 0; i < dim * dim; ++i) {
    EXPECT_EQ(expected[i], out[i]);
  }
}
template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
void EigenBackendMatmul::test_square_batch_matmul(std::vector<T>& lhs,
                                                  std::vector<T>& rhs,
                                                  std::vector<T>& expected,
                                                  Index batch, Index dim) {
  auto device = get_eigen_device();
  const auto size = batch * dim * dim * sizeof(T);
  T* lhs_ptr = backend_.allocate<T>(size);
  T* rhs_ptr = backend_.allocate<T>(size);
  T* out_ptr = backend_.allocate<T>(size);

  device.memcpyHostToDevice(lhs_ptr, lhs.data(), size);
  device.memcpyHostToDevice(rhs_ptr, rhs.data(), size);

  backend_.batch_matmul<TransposeLHS, TransposeRHS>(lhs_ptr, rhs_ptr, out_ptr,
                                                    batch, dim, dim, dim);
  std::vector<T> out(batch * dim * dim);
  device.memcpyDeviceToHost(out.data(), out_ptr, size);
  for (int i = 0; i < dim * dim; ++i) {
    EXPECT_EQ(expected[i], out[i]);
  }
}
#endif  // SYCLDNN_TEST_BACKEND_EIGEN_MATMUL_BACKEND_TEST_FIXTURE_IMPL_H_
