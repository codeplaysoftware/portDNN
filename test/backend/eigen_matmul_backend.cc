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
#include <gtest/gtest.h>
#include "test/backend/eigen_matmul_backend_test_fixture.h"
#include "test/types/kernel_data_types.h"

namespace {
cl::sycl::default_selector selector{};
}  // namespace
std::unique_ptr<Eigen::QueueInterface> EigenBackendTest::queue_interface_{
    new Eigen::QueueInterface{selector}};
Eigen::SyclDevice EigenBackendTest::device_{
    EigenBackendTest::queue_interface_.get()};
sycldnn::backend::EigenBackend EigenBackendTest::backend_{
    EigenBackendTest::device_};

template <typename T>
using EigenMatmul = EigenBackendMatmul;
using MatmulTypes = sycldnn::types::GTestKernelDataTypes;
TYPED_TEST_CASE(EigenMatmul, MatmulTypes);

TYPED_TEST(EigenMatmul, SimpleMatmul) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4};
  std::vector<TypeParam> expected = {7, 10, 15, 22};
  this->template test_square_matmul<false, false>(lhs, rhs, expected, 2);
}
TYPED_TEST(EigenMatmul, SimpleBatchMatmul) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> expected = {7,  10, 15, 22, 7,  10,
                                     15, 22, 7,  10, 15, 22};
  this->template test_square_batch_matmul<false, false>(lhs, rhs, expected, 3,
                                                        2);
}
TYPED_TEST(EigenMatmul, SimpleMatmulTlhs) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4};
  std::vector<TypeParam> expected = {10, 14, 14, 20};
  this->template test_square_matmul<true, false>(lhs, rhs, expected, 2);
}
TYPED_TEST(EigenMatmul, SimpleBatchMatmulTlhs) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> expected = {10, 14, 14, 20, 10, 14,
                                     14, 20, 10, 14, 14, 20};
  this->template test_square_batch_matmul<true, false>(lhs, rhs, expected, 3,
                                                       2);
}
TYPED_TEST(EigenMatmul, SimpleMatmulTrhs) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4};
  std::vector<TypeParam> expected = {5, 11, 11, 25};
  this->template test_square_matmul<false, true>(lhs, rhs, expected, 2);
}
TYPED_TEST(EigenMatmul, SimpleBatchMatmulTrhs) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> expected = {5,  11, 11, 25, 5,  11,
                                     11, 25, 5,  11, 11, 25};
  this->template test_square_batch_matmul<false, true>(lhs, rhs, expected, 3,
                                                       2);
}
TYPED_TEST(EigenMatmul, SimpleMatmulTlhsTrhs) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4};
  std::vector<TypeParam> expected = {7, 15, 10, 22};
  this->template test_square_matmul<true, true>(lhs, rhs, expected, 2);
}
TYPED_TEST(EigenMatmul, SimpleBatchMatmulTlhsTrhs) {
  std::vector<TypeParam> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> rhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<TypeParam> expected = {7,  15, 10, 22, 7,  15,
                                     10, 22, 7,  15, 10, 22};
  this->template test_square_batch_matmul<true, true>(lhs, rhs, expected, 3, 2);
}
