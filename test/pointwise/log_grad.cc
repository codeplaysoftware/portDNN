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
// This file was automatically generated by generate_pointwise_tests.py.
// Results calculated using Tensorflow v2.11.0.

#include <gtest/gtest.h>

#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/operators.h"

#include "test/pointwise/pointwise_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"

using namespace sycldnn;  // NOLINT(google-build-using-namespace)

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

template <typename DataType>
using LogGrad = PointwiseFixture<DataType, pointwise::Log, pointwise::Gradient>;
TYPED_TEST_SUITE(LogGrad, GTestTypePairs);
TYPED_TEST(LogGrad, Shape_1x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> input = iota_initialised_data<DataType>(1, 1);
  const std::vector<DataType> exp_out = {1.};
  this->test_pointwise(input, exp_out);
}
TYPED_TEST(LogGrad, Shape_8x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> input = iota_initialised_data<DataType>(8, 8);
  const std::vector<DataType> exp_out = {1., 1., 1., 1., 1., 1., 1., 1.};
  this->test_pointwise(input, exp_out);
}
TYPED_TEST(LogGrad, Shape_9x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> input = iota_initialised_data<DataType>(9, 9);
  const std::vector<DataType> exp_out = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
  this->test_pointwise(input, exp_out);
}
TYPED_TEST(LogGrad, Shape_10x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> input = iota_initialised_data<DataType>(10, 10);
  const std::vector<DataType> exp_out = {1., 1., 1., 1., 1.,
                                         1., 1., 1., 1., 1.};
  this->test_pointwise(input, exp_out);
}
