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
#ifndef PORTDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_H_
#define PORTDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_H_

#include "test/backend/backend_test_fixture.h"

template <typename Backend>
struct BackendMatmul : public BackendTestFixture<Backend> {
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_nonsquare_matmul(std::vector<T> const& lhs,
                             std::vector<T> const& rhs,
                             std::vector<T> const& expected, Index m, Index n,
                             Index k);
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_square_matmul(std::vector<T> const& lhs, std::vector<T> const& rhs,
                          std::vector<T> const& expected, Index dim);
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  void test_square_batch_matmul(std::vector<T> const& lhs,
                                std::vector<T> const& rhs,
                                std::vector<T> const& expected, Index batch,
                                Index dim);
};

#endif  // PORTDNN_TEST_BACKEND_MATMUL_BACKEND_TEST_FIXTURE_H_
