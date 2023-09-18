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

#include "portdnn/data_format.h"

#include "portdnn/softmax/direction.h"
#include "portdnn/softmax/params.h"

#include "test/softmax/softmax_event_dependencies_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_triple.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>

using DataTypeList = sycldnn::types::KernelDataTypes;
using DataFormats = sycldnn::types::DataFormatTypes;

using TypeFormatPairs =
    sycldnn::types::CartesianProduct<DataTypeList, DataFormats>::type;

using GTestTypePair = sycldnn::types::ToGTestTypes<TypeFormatPairs>::type;

using namespace sycldnn;  // NOLINT(google-build-using-namespace)
template <typename Pair>
using SoftmaxForward = SoftmaxEventFixture<Pair, softmax::Forward>;
TYPED_TEST_CASE(SoftmaxForward, GTestTypePair);
TYPED_TEST(SoftmaxForward, 1x1x1x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 1, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x1x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 1, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x1x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 1, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x8x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 8, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x8x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 8, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x8x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 8, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x9x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 9, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x9x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 9, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x1x9x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 1, 9, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x1x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 1, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x1x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 1, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x1x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 1, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x8x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 8, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x8x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 8, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x8x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 8, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x9x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 9, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x9x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 9, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x8x9x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 8, 9, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x1x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 1, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x1x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 1, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x1x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 1, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x8x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 8, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x8x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 8, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x8x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 8, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x9x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 9, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x9x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 9, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 1x9x9x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{1, 9, 9, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x1x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 1, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x1x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 1, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x1x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 1, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x8x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 8, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x8x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 8, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x8x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 8, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x9x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 9, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x9x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 9, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x1x9x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 1, 9, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x1x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 1, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x1x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 1, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x1x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 1, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x8x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 8, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x8x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 8, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x8x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 8, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x9x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 9, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x9x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 9, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x8x9x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 8, 9, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x1x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 1, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x1x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 1, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x1x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 1, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x8x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 8, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x8x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 8, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x8x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 8, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x9x1) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 9, 1}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x9x5) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 9, 5}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
TYPED_TEST(SoftmaxForward, 3x9x9x8) {
  using DataType = typename TestFixture::DataType;
  const std::array<int, 4> in_shape = {{3, 9, 9, 8}};
  const auto params = getSoftmaxParams(in_shape);
  const DataType max_input_val = 2048.0;
  this->test_softmax(params, max_input_val);
}
