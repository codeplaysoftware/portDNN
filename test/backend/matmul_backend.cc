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

#ifdef SNN_TEST_EIGEN
// TODO(jwlawson): remove cassert when no longer needed before Eigen include
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sycldnn/backend/eigen_backend.h"
#include "sycldnn/backend/eigen_backend_with_snn_matmul.h"
#endif  // SNN_TEST_EIGEN

#ifdef SNN_TEST_SYCLBLAS
#include "sycldnn/backend/sycl_blas_backend.h"
#endif  // SNN_TEST_SYCLBLAS

#include "test/backend/matmul_backend_test_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/concatenate.h"
#include "test/types/kernel_data_types.h"

template <typename TypePair>
using Matmul = BackendMatmul<typename TypePair::FirstType>;
using MatmulTypes = sycldnn::types::GTestKernelDataTypes;

// List the enabled backends.
using EigenBackends = sycldnn::types::TypeList<
#ifdef SNN_TEST_EIGEN
    sycldnn::backend::EigenBackendSNNMatmul, sycldnn::backend::EigenBackend
#endif
    >;
using SyclblasBackends = sycldnn::types::TypeList<
#ifdef SNN_TEST_SYCLBLAS
    sycldnn::backend::SyclBLASBackend
#endif
    >;
using Backends =
    sycldnn::types::Concatenate<EigenBackends, SyclblasBackends>::type;

// List the supported data types.
using DataTypeList = sycldnn::types::KernelDataTypes;

// The list of backend/data type pairs.
using TestTypePairs =
    typename sycldnn::types::CartesianProduct<Backends, DataTypeList>::type;

// GTest compatible type list
using GTestTypePairs = sycldnn::types::ToGTestTypes<TestTypePairs>::type;
TYPED_TEST_CASE(Matmul, GTestTypePairs);

TYPED_TEST(Matmul, SimpleMatmul) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4};
  std::vector<DataType> rhs = {5, 6, 7, 8};
  std::vector<DataType> expected = {19, 22, 43, 50};
  this->template test_square_matmul<false, false>(lhs, rhs, expected, 2);
}
TYPED_TEST(Matmul, SimpleMatmulNonSquare) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3};
  std::vector<DataType> rhs = {4, 5, 6, 7, 8, 9};
  std::vector<DataType> expected = {40, 46};
  this->template test_nonsquare_matmul<false, false>(lhs, rhs, expected, 1, 2,
                                                     3);
}
TYPED_TEST(Matmul, SimpleMatmulNonSquare2) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4, 5, 6};
  std::vector<DataType> rhs = {1, 2, 3, 4, 5, 6};
  std::vector<DataType> expected = {22, 28, 49, 64};
  this->template test_nonsquare_matmul<false, false>(lhs, rhs, expected, 2, 2,
                                                     3);
}
TYPED_TEST(Matmul, SimpleBatchMatmul) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<DataType> rhs = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
  std::vector<DataType> expected = {7,  10, 15, 22, 19, 22,
                                    43, 50, 23, 34, 31, 46};
  this->template test_square_batch_matmul<false, false>(lhs, rhs, expected, 3,
                                                        2);
}
TYPED_TEST(Matmul, SimpleMatmulTlhs) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4};
  std::vector<DataType> rhs = {5, 6, 7, 8};
  std::vector<DataType> expected = {26, 30, 38, 44};
  this->template test_square_matmul<true, false>(lhs, rhs, expected, 2);
}
TYPED_TEST(Matmul, SimpleBatchMatmulTlhs) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<DataType> rhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<DataType> expected = {10, 14, 14, 20, 10, 14,
                                    14, 20, 10, 14, 14, 20};
  this->template test_square_batch_matmul<true, false>(lhs, rhs, expected, 3,
                                                       2);
}
TYPED_TEST(Matmul, SimpleMatmulTrhs) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4};
  std::vector<DataType> rhs = {1, 2, 3, 4};
  std::vector<DataType> expected = {5, 11, 11, 25};
  this->template test_square_matmul<false, true>(lhs, rhs, expected, 2);
}
TYPED_TEST(Matmul, SimpleBatchMatmulTrhs) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<DataType> rhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<DataType> expected = {5,  11, 11, 25, 5,  11,
                                    11, 25, 5,  11, 11, 25};
  this->template test_square_batch_matmul<false, true>(lhs, rhs, expected, 3,
                                                       2);
}
TYPED_TEST(Matmul, SimpleMatmulTlhsTrhs) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4};
  std::vector<DataType> rhs = {1, 2, 3, 4};
  std::vector<DataType> expected = {7, 15, 10, 22};
  this->template test_square_matmul<true, true>(lhs, rhs, expected, 2);
}
TYPED_TEST(Matmul, SimpleBatchMatmulTlhsTrhs) {
  using DataType = typename TypeParam::SecondType;
  std::vector<DataType> lhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<DataType> rhs = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  std::vector<DataType> expected = {7,  15, 10, 22, 7,  15,
                                    10, 22, 7,  15, 10, 22};
  this->template test_square_batch_matmul<true, true>(lhs, rhs, expected, 3, 2);
}
