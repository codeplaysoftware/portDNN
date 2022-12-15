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
// This file was automatically generated by generate_gather_tests.py.
// Results calculated using Tensorflow v2.8.0.

#include <gtest/gtest.h>

#include <vector>

#include "test/gather/gather_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes_;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using namespace sycldnn;
using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;
using IndexDataType = int32_t;  // or int64_t

template <typename Pair>
using Gather2D = GatherFixture<Pair, IndexDataType>;
TYPED_TEST_SUITE(Gather2D, GTestTypePairs);
TYPED_TEST(Gather2D, G2D_Axis_Neg2_Inp5x4_Ind1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {9., 10., 11., 12.};
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_Neg2_Inp5x4_Ind5) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {17., 18., 19., 20., 9.,  10., 11.,
                                         12., 5.,  6.,  7.,  8.,  13., 14.,
                                         15., 16., 9.,  10., 11., 12.};
  const std::vector<IndexDataType> indices = {4, 2, 1, 3, 2};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {5};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_Neg2_Inp5x4_Ind5x2) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      13., 14., 15., 16., 5., 6., 7., 8., 5.,  6.,  7.,  8., 1., 2.,
      3.,  4.,  5.,  6.,  7., 8., 5., 6., 7.,  8.,  1.,  2., 3., 4.,
      1.,  2.,  3.,  4.,  5., 6., 7., 8., 13., 14., 15., 16.};
  const std::vector<IndexDataType> indices = {3, 1, 1, 0, 1, 1, 0, 0, 1, 3};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {5, 2};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_Neg1_Inp5x4_Ind1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {3., 7., 11., 15., 19.};
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_Neg1_Inp5x4_Ind4) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {2.,  1.,  1.,  1.,  6.,  5.,  5.,
                                         5.,  10., 9.,  9.,  9.,  14., 13.,
                                         13., 13., 18., 17., 17., 17.};
  const std::vector<IndexDataType> indices = {1, 0, 0, 0};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {4};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_Neg1_Inp5x4_Ind4x2) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      1.,  2.,  4.,  4.,  3.,  1.,  4.,  3.,  5.,  6.,  8.,  8.,  7.,  5.,
      8.,  7.,  9.,  10., 12., 12., 11., 9.,  12., 11., 13., 14., 16., 16.,
      15., 13., 16., 15., 17., 18., 20., 20., 19., 17., 20., 19.};
  const std::vector<IndexDataType> indices = {0, 1, 3, 3, 2, 0, 3, 2};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {4, 2};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_0_Inp5x4_Ind1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {17., 18., 19., 20.};
  const std::vector<IndexDataType> indices = {4};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_0_Inp5x4_Ind5) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1.,  2.,  3.,  4.,  1.,  2.,  3.,
                                         4.,  5.,  6.,  7.,  8.,  13., 14.,
                                         15., 16., 17., 18., 19., 20.};
  const std::vector<IndexDataType> indices = {0, 0, 1, 3, 4};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {5};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_0_Inp5x4_Ind5x2) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      17., 18., 19., 20., 17., 18., 19., 20., 5., 6.,  7.,  8.,  13., 14.,
      15., 16., 9.,  10., 11., 12., 5.,  6.,  7., 8.,  17., 18., 19., 20.,
      1.,  2.,  3.,  4.,  13., 14., 15., 16., 9., 10., 11., 12.};
  const std::vector<IndexDataType> indices = {4, 4, 1, 3, 2, 1, 4, 0, 3, 2};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {5, 2};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_1_Inp5x4_Ind1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {2., 6., 10., 14., 18.};
  const std::vector<IndexDataType> indices = {1};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_1_Inp5x4_Ind4) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1.,  4.,  3.,  3.,  5.,  8.,  7.,
                                         7.,  9.,  12., 11., 11., 13., 16.,
                                         15., 15., 17., 20., 19., 19.};
  const std::vector<IndexDataType> indices = {0, 3, 2, 2};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {4};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
TYPED_TEST(Gather2D, G2D_Axis_1_Inp5x4_Ind4x2) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      3.,  3.,  2.,  3.,  1.,  4.,  4.,  4.,  7.,  7.,  6.,  7.,  5.,  8.,
      8.,  8.,  11., 11., 10., 11., 9.,  12., 12., 12., 15., 15., 14., 15.,
      13., 16., 16., 16., 19., 19., 18., 19., 17., 20., 20., 20.};
  const std::vector<IndexDataType> indices = {2, 2, 1, 2, 0, 3, 3, 3};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {4, 2};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
