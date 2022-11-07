/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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

// DO NOT MODIFY BY HAND
// This file was automatically generated by generate_transpose_tests.py.
// Results calculated using Tensorflow v2.8.0.

#include <gtest/gtest.h>
#include <vector>

#include "test/transpose/transpose_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes_;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

template <typename Pair>
using Transpose2D = TransposeFixture<Pair>;
TYPED_TEST_SUITE(Transpose2D, GTestTypePairs);
TYPED_TEST(Transpose2D, T2D_2x2_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4.};
  const std::vector<int> sizes = {2, 2};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_2x2_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 3., 2., 4.};
  const std::vector<int> sizes = {2, 2};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_2x3_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5., 6.};
  const std::vector<int> sizes = {2, 3};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_2x3_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 4., 2., 5., 3., 6.};
  const std::vector<int> sizes = {2, 3};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_2x4_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5., 6., 7., 8.};
  const std::vector<int> sizes = {2, 4};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_2x4_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 5., 2., 6., 3., 7., 4., 8.};
  const std::vector<int> sizes = {2, 4};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_3x2_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5., 6.};
  const std::vector<int> sizes = {3, 2};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_3x2_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 3., 5., 2., 4., 6.};
  const std::vector<int> sizes = {3, 2};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_3x3_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  const std::vector<int> sizes = {3, 3};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_3x3_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 4., 7., 2., 5., 8., 3., 6., 9.};
  const std::vector<int> sizes = {3, 3};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_3x4_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4.,  5.,  6.,
                                         7., 8., 9., 10., 11., 12.};
  const std::vector<int> sizes = {3, 4};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_3x4_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 5., 9.,  2., 6., 10.,
                                         3., 7., 11., 4., 8., 12.};
  const std::vector<int> sizes = {3, 4};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_4x2_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5., 6., 7., 8.};
  const std::vector<int> sizes = {4, 2};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_4x2_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 3., 5., 7., 2., 4., 6., 8.};
  const std::vector<int> sizes = {4, 2};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_4x3_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4.,  5.,  6.,
                                         7., 8., 9., 10., 11., 12.};
  const std::vector<int> sizes = {4, 3};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_4x3_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 4.,  7., 10., 2., 5.,
                                         8., 11., 3., 6.,  9., 12.};
  const std::vector<int> sizes = {4, 3};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_4x4_0x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2.,  3.,  4.,  5.,  6.,  7.,  8.,
                                         9., 10., 11., 12., 13., 14., 15., 16.};
  const std::vector<int> sizes = {4, 4};
  const std::vector<int> perm = {0, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
TYPED_TEST(Transpose2D, T2D_4x4_1x0) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 5., 9.,  13., 2., 6., 10., 14.,
                                         3., 7., 11., 15., 4., 8., 12., 16.};
  const std::vector<int> sizes = {4, 4};
  const std::vector<int> perm = {1, 0};
  const DataType max_input_val = 2048.0;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
