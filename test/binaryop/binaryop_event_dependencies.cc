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
#include <vector>

#include "portdnn/binaryop/operators.h"
#include "test/binaryop/binaryop_event_dependencies_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

using DataTypeList = sycldnn::types::KernelDataTypes;

using GTestTypeList = sycldnn::types::ToGTestTypes<DataTypeList>::type;

template <typename DataType>
using BinaryAdd = BinaryOpEventFixture<DataType, sycldnn::binaryop::Add>;
TYPED_TEST_SUITE(BinaryAdd, GTestTypeList);
TYPED_TEST(BinaryAdd, lhs_1_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 1;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_1_rhs_12) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {12};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_12_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {12};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_1_3_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 3;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 3};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_2_3_4_5_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_2_3_4_5_rhs_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_4_5_rhs_2_3_4_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {4, 5};
  params.rhs_dims = {2, 3, 4, 5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_1_4_5_rhs_2_3_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 4, 5};
  params.rhs_dims = {2, 3, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryAdd, lhs_3_4_5_rhs_2_1_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {3, 4, 5};
  params.rhs_dims = {2, 1, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}

template <typename DataType>
using BinaryDiv = BinaryOpEventFixture<DataType, sycldnn::binaryop::Div>;
TYPED_TEST_SUITE(BinaryDiv, GTestTypeList);
TYPED_TEST(BinaryDiv, lhs_1_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 1;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_1_rhs_12) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {12};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_12_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {12};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_1_3_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 3;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 3};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_2_3_4_5_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_2_3_4_5_rhs_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_4_5_rhs_2_3_4_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {4, 5};
  params.rhs_dims = {2, 3, 4, 5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_1_4_5_rhs_2_3_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 4, 5};
  params.rhs_dims = {2, 3, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryDiv, lhs_3_4_5_rhs_2_1_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {3, 4, 5};
  params.rhs_dims = {2, 1, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}

template <typename DataType>
using BinaryMul = BinaryOpEventFixture<DataType, sycldnn::binaryop::Mul>;
TYPED_TEST_SUITE(BinaryMul, GTestTypeList);
TYPED_TEST(BinaryMul, lhs_1_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 1;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_1_rhs_12) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {12};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_12_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {12};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_1_3_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 3;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 3};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_2_3_4_5_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_2_3_4_5_rhs_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_4_5_rhs_2_3_4_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {4, 5};
  params.rhs_dims = {2, 3, 4, 5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_1_4_5_rhs_2_3_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 4, 5};
  params.rhs_dims = {2, 3, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinaryMul, lhs_3_4_5_rhs_2_1_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {3, 4, 5};
  params.rhs_dims = {2, 1, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}

template <typename DataType>
using BinarySub = BinaryOpEventFixture<DataType, sycldnn::binaryop::Sub>;
TYPED_TEST_SUITE(BinarySub, GTestTypeList);
TYPED_TEST(BinarySub, lhs_1_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 1;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_1_rhs_12) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1};
  params.rhs_dims = {12};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_12_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 12;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {12};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_1_3_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 3;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 3};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_2_3_4_5_rhs_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_2_3_4_5_rhs_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {2, 3, 4, 5};
  params.rhs_dims = {5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_4_5_rhs_2_3_4_5) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {4, 5};
  params.rhs_dims = {2, 3, 4, 5};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_1_4_5_rhs_2_3_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {1, 4, 5};
  params.rhs_dims = {2, 3, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
TYPED_TEST(BinarySub, lhs_3_4_5_rhs_2_1_1_1) {
  using DataType = typename TestFixture::DataType;
  const unsigned int exp_size = 120;
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {3, 4, 5};
  params.rhs_dims = {2, 1, 1, 1};
  const DataType max_input_val = 2048.0;
  this->run(exp_size, params, max_input_val);
}
