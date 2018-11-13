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
#ifndef SYCLDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_H_
#define SYCLDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_H_

#include "test/backend/backend_test_fixture.h"

template <typename Backend>
struct BackendMatmul : public BackendTest<Backend> {
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_nonsquare_matmul(std::vector<T>& lhs, std::vector<T>& rhs,
                             std::vector<T>& expected, Index m, Index n,
                             Index k);
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_square_matmul(std::vector<T>& lhs, std::vector<T>& rhs,
                          std::vector<T>& expected, Index dim);
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_square_batch_matmul(std::vector<T>& lhs, std::vector<T>& rhs,
                                std::vector<T>& expected, Index batch,
                                Index dim);
  template <typename T>
  void copy_to_device(T* src, T* dst, size_t nelems);
  template <typename T>
  void copy_to_host(T* src, T* dst, size_t nelems);
};

#endif  // SYCLDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_H_
