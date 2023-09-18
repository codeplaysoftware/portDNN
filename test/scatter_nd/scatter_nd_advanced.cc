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

#include <gtest/gtest.h>

#include "portdnn/scatter_nd/operators.h"
#include "portdnn/scatter_nd/params.h"

#include "test/scatter_nd/scatter_nd_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

using namespace sycldnn;  // NOLINT(google-build-using-namespace)
template <typename Pair>
using ScatterNDAssign = ScatterNDFixture<Pair, int, scatter_nd::Assign>;
TYPED_TEST_CASE(ScatterNDAssign, GTestTypePairs);
TYPED_TEST(ScatterNDAssign, 1x1x1x8_elementwise_neg_ind) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {0, 1, 0, 0, 2, 0, 0, 0};
  const std::array<int, 4> in_shape = {{8, 1, 1, 1}};
  const std::array<int, 2> ind_shape = {2, 1};
  const auto params = getScatterNDParams(in_shape, ind_shape);
  const std::vector<DataType> input = {0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<int> indices = {-7, -4};
  const std::vector<DataType> updates = {1, 2};
  this->test_scatter_nd(input, indices, updates, exp_out, params);
}

TYPED_TEST(ScatterNDAssign, 2x2x2x2_vector_slice_out_of_bounds) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {0, 0, 1, 2, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0};
  const std::array<int, 4> in_shape = {{2, 2, 2, 2}};
  const std::array<int, 2> ind_shape = {3, 3};
  const auto params = getScatterNDParams(in_shape, ind_shape);
  const std::vector<DataType> input = {0, 0, 0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<int> indices = {0, 0, 2, -3, 0, 0, 0, 0, 1};
  const std::vector<DataType> updates = {-1, -1, -3, -5, 1, 2};
  this->test_scatter_nd(input, indices, updates, exp_out, params);
}

template <typename DataType>
using ScatterNDSub = ScatterNDFixture<DataType, int, scatter_nd::Sub>;
TYPED_TEST_CASE(ScatterNDSub, GTestTypePairs);
TYPED_TEST(ScatterNDSub, 3x1x8x1_matrix_slice_Sub) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {2,  -3, 7, 8,  5,  0,  -5, -9,
                                         2,  4,  0, 8,  4,  6,  2,  3,
                                         -1, 0,  1, -4, -2, -3, -5, -2};
  const std::array<int, 4> in_shape = {{3, 1, 8, 1}};
  const std::array<int, 2> ind_shape = {2, 2};
  const auto params = getScatterNDParams(in_shape, ind_shape);
  const std::vector<DataType> input = {7, 2, 9, 9, 7, 6, 1, 0, 2, 4, 0, 8,
                                       4, 6, 2, 3, 1, 7, 7, 2, 7, 0, 3, 1};
  const std::vector<int> indices = {2, 0, 0, 0};
  const std::vector<DataType> updates = {2, 7, 6, 6, 9, 3, 8, 3,
                                         5, 5, 2, 1, 2, 6, 6, 9};
  this->test_scatter_nd(input, indices, updates, exp_out, params);
}

template <typename DataType>
using ScatterNDMul = ScatterNDFixture<DataType, int, scatter_nd::Mul>;
TYPED_TEST_CASE(ScatterNDMul, GTestTypePairs);
TYPED_TEST(ScatterNDMul, 1x1x2x5_tensor_slice_Mul) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {48, 6, 35, 15, 4, 7, 12, 36, 54, 27};
  const std::array<int, 4> in_shape = {{1, 1, 2, 5}};
  const std::array<int, 2> ind_shape = {1, 1};
  const auto params = getScatterNDParams(in_shape, ind_shape);
  const std::vector<DataType> input = {8, 3, 5, 5, 2, 1, 2, 6, 6, 9};
  const std::vector<int> indices = {0};
  const std::vector<DataType> updates = {6, 2, 7, 3, 2, 7, 6, 6, 9, 3};
  this->test_scatter_nd(input, indices, updates, exp_out, params);
}

template <typename DataType>
using ScatterNDDiv = ScatterNDFixture<DataType, int, scatter_nd::Div>;
TYPED_TEST_CASE(ScatterNDDiv, GTestTypePairs);
TYPED_TEST(ScatterNDDiv, 1x1x1x8_elementwise_Div) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {5, 2, 2, 1, 8, 6, 6, 3};
  const std::array<int, 4> in_shape = {{1, 1, 1, 8}};
  const std::array<int, 2> ind_shape = {4, 4};
  const auto params = getScatterNDParams(in_shape, ind_shape);
  const std::vector<DataType> input = {5, 4, 6, 1, 4, 6, 6, 9};
  const std::vector<int> indices = {0, 0, 0, 1, 0, 0, 0, 2,
                                    0, 0, 0, 4, 0, 0, 0, 7};
  const std::vector<DataType> updates = {2, 3, 0.5, 3};
  this->test_scatter_nd(input, indices, updates, exp_out, params);
}
