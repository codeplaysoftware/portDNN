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

#include "portdnn/padding_mode.h"

#include "portdnn/pooling/operators.h"

#include "test/pooling/pooling_fixture.h"

#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_triple.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"
#include "test/types/type_list.h"

#include <array>
#include <vector>

template <typename Triple>
using OneByOneInputAvgPoolingTest =
    PoolingFixture<typename Triple::FirstType, typename Triple::SecondType,
                   sycldnn::backend::SNNBackend, sycldnn::pooling::Average,
                   typename Triple::ThirdType>;

using DataTypeList = sycldnn::types::KernelDataTypes;
// NB - Pooling reduces the input in its spatial dimensions. Since our
// inputs are degenerate 1x1 cases, forward and backward passes will
// have the same results, when they normally wouldn't.
using Directions = sycldnn::types::TypeList<sycldnn::pooling::Forward,
                                            sycldnn::pooling::Backpropagate>;
using DataFormatLists = sycldnn::types::DataFormatTypes;
using SNNTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, DataFormatLists>::type;
using DirectedBackTypePairs =
    sycldnn::types::CartesianProduct<SNNTypePairs, Directions>::type;
using TestTriples =
    sycldnn::types::NestedPairsToTriple<DirectedBackTypePairs>::type;
using GTestTypeTriples = sycldnn::types::ToGTestTypes<TestTriples>::type;

TYPED_TEST_SUITE(OneByOneInputAvgPoolingTest, GTestTypeTriples);

/*
 * Input: 1    Output: 1
 */
TYPED_TEST(OneByOneInputAvgPoolingTest, Basic1x1Plain) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<1, 1>(in_shape, padding);
  const DataType max_input_val = 1.0;
  this->test_pool(exp_out, params, max_input_val);
}

/*
 * Input: 1       Output: 1
 *         2               2
 *          3               3
 *           4               4
 */
TYPED_TEST(OneByOneInputAvgPoolingTest, Deep1x1Plain) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 4}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<1, 1>(in_shape, padding);
  const DataType max_input_val = 4.0;
  this->test_pool(exp_out, params, max_input_val);
}

/*
 * Input: 1  5     Output: 1  5
 *         2  6             2  6
 *          3  7             3  7
 *           4  8             4  8
 */
TYPED_TEST(OneByOneInputAvgPoolingTest, BatchedDeep1x1Plain) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5., 6., 7., 8.};
  const std::array<int, 4> in_shape = {{2, 1, 1, 4}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<1, 1>(in_shape, padding);
  const DataType max_input_val = 8.0;
  this->test_pool(exp_out, params, max_input_val);
}

/*
 * Input: 1  5     Output: 1  5
 *         2  6             2  6
 *          3  7             3  7
 *           4  8             4  8
 */
TYPED_TEST(OneByOneInputAvgPoolingTest, BatchedDeep1x1Plain2x2Window) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4., 5., 6., 7., 8.};
  const std::array<int, 4> in_shape = {{2, 1, 1, 4}};
  const auto padding = sycldnn::PaddingMode::SAME;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  const DataType max_input_val = 8.0;
  this->test_pool(exp_out, params, max_input_val);
}
