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
#ifndef SYCLDNN_TEST_BACKEND_EIGEN_MATMUL_BACKEND_TEST_FIXTURE_H_
#define SYCLDNN_TEST_BACKEND_EIGEN_MATMUL_BACKEND_TEST_FIXTURE_H_
#include "test/backend/eigen_backend_test_fixture.h"

struct EigenBackendMatmul : public EigenBackendTest {
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_square_matmul(std::vector<T>& lhs, std::vector<T>& rhs,
                          std::vector<T>& expected, Index dim);
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_square_batch_matmul(std::vector<T>& lhs, std::vector<T>& rhs,
                                std::vector<T>& expected, Index batch,
                                Index dim);
};
#endif  // SYCLDNN_TEST_BACKEND_EIGEN_MATMUL_BACKEND_TEST_FIXTURE_H_
