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

#include "test/gather/gather_event_dependencies_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

using IntDataTypeList =
    sycldnn::types::TypeList<int8_t, int16_t, int32_t, int64_t, uint8_t,
                             uint16_t, uint32_t, uint64_t>;
using DataTypeList =
    sycldnn::types::Concatenate<sycldnn::types::KernelDataTypes,
                                IntDataTypeList>::type;

using namespace sycldnn;
using GTestTypeList = sycldnn::types::ToGTestTypes<DataTypeList>::type;
using IndexDataType = int32_t;  // or int64_t

template <typename DType>
using GatherEvent = GatherEventFixture<DType, IndexDataType>;
TYPED_TEST_SUITE(GatherEvent, GTestTypeList);
TYPED_TEST(GatherEvent, G1D_Axis_Neg1_Inp5_Ind1) {
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {1};
  params.input_dims = {5};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G1D_Axis_Neg1_Inp5_Ind5) {
  const std::vector<IndexDataType> indices = {4, 2, 1, 3, 2};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {5};
  params.input_dims = {5};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G1D_Axis_Neg1_Inp5_Ind5x2) {
  const std::vector<IndexDataType> indices = {3, 1, 1, 0, 1, 1, 0, 0, 1, 3};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {5, 2};
  params.input_dims = {5};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G1D_Axis_0_Inp5_Ind1) {
  const std::vector<IndexDataType> indices = {4};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {1};
  params.input_dims = {5};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G1D_Axis_0_Inp5_Ind5) {
  const std::vector<IndexDataType> indices = {0, 0, 4, 1, 3};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {5};
  params.input_dims = {5};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G1D_Axis_0_Inp5_Ind5x2) {
  const std::vector<IndexDataType> indices = {2, 4, 2, 4, 0, 0, 1, 3, 4, 4};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {5, 2};
  params.input_dims = {5};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G2D_Axis_Neg2_Inp5x4_Ind1) {
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G2D_Axis_Neg2_Inp5x4_Ind5) {
  const std::vector<IndexDataType> indices = {4, 2, 1, 3, 2};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {5};
  params.input_dims = {5, 4};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G2D_Axis_Neg2_Inp5x4_Ind5x2) {
  const std::vector<IndexDataType> indices = {3, 1, 1, 0, 1, 1, 0, 0, 1, 3};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {5, 2};
  params.input_dims = {5, 4};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G2D_Axis_Neg1_Inp5x4_Ind1) {
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G2D_Axis_1_Inp5x4_Ind4) {
  const std::vector<IndexDataType> indices = {0, 3, 2, 2};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {4};
  params.input_dims = {5, 4};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G2D_Axis_1_Inp5x4_Ind4x2) {
  const std::vector<IndexDataType> indices = {2, 2, 1, 2, 0, 3, 3, 3};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {4, 2};
  params.input_dims = {5, 4};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg3_Inp5x4x3_Ind1) {
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = -3;
  params.indices_dims = {1};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg3_Inp5x4x3_Ind5) {
  const std::vector<IndexDataType> indices = {4, 2, 1, 3, 2};
  gather::GatherParams params;
  params.axis = -3;
  params.indices_dims = {5};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg3_Inp5x4x3_Ind5x2) {
  const std::vector<IndexDataType> indices = {3, 1, 1, 0, 1, 1, 0, 0, 1, 3};
  gather::GatherParams params;
  params.axis = -3;
  params.indices_dims = {5, 2};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg2_Inp5x4x3_Ind1) {
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {1};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg2_Inp5x4x3_Ind4) {
  const std::vector<IndexDataType> indices = {1, 0, 0, 0};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {4};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg2_Inp5x4x3_Ind4x2) {
  const std::vector<IndexDataType> indices = {0, 1, 3, 3, 2, 0, 3, 2};
  gather::GatherParams params;
  params.axis = -2;
  params.indices_dims = {4, 2};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg1_Inp5x4x3_Ind1) {
  const std::vector<IndexDataType> indices = {0};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {1};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg1_Inp5x4x3_Ind3) {
  const std::vector<IndexDataType> indices = {0, 1, 0};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {3};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_Neg1_Inp5x4x3_Ind3x2) {
  const std::vector<IndexDataType> indices = {1, 0, 0, 0, 2, 1};
  gather::GatherParams params;
  params.axis = -1;
  params.indices_dims = {3, 2};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_0_Inp5x4x3_Ind1) {
  const std::vector<IndexDataType> indices = {3};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {1};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_0_Inp5x4x3_Ind5) {
  const std::vector<IndexDataType> indices = {2, 1, 4, 0, 3};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {5};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_0_Inp5x4x3_Ind5x2) {
  const std::vector<IndexDataType> indices = {2, 0, 3, 2, 2, 2, 2, 4, 3, 3};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {5, 2};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_1_Inp5x4x3_Ind1) {
  const std::vector<IndexDataType> indices = {1};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {1};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_1_Inp5x4x3_Ind4) {
  const std::vector<IndexDataType> indices = {0, 2, 0, 2};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {4};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_1_Inp5x4x3_Ind4x2) {
  const std::vector<IndexDataType> indices = {1, 3, 2, 0, 2, 2, 3, 0};
  gather::GatherParams params;
  params.axis = 1;
  params.indices_dims = {4, 2};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_2_Inp5x4x3_Ind1) {
  const std::vector<IndexDataType> indices = {2};
  gather::GatherParams params;
  params.axis = 2;
  params.indices_dims = {1};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_2_Inp5x4x3_Ind3) {
  const std::vector<IndexDataType> indices = {1, 2, 1};
  gather::GatherParams params;
  params.axis = 2;
  params.indices_dims = {3};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
TYPED_TEST(GatherEvent, G3D_Axis_2_Inp5x4x3_Ind3x2) {
  const std::vector<IndexDataType> indices = {1, 2, 0, 0, 1, 0};
  gather::GatherParams params;
  params.axis = 2;
  params.indices_dims = {3, 2};
  params.input_dims = {5, 4, 3};
  this->test_gather(params, indices);
}
