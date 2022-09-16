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

#include "test/gather/gather_fixture.h"
#include "test/types/concatenate.h"
#include "test/types/kernel_data_types.h"
#include "test/types/to_gtest_types.h"

using namespace sycldnn;

using GTestTypeList = sycldnn::types::GTestKernelDataTypes;
using IndexDataType = int32_t;  // or int64_t

template <typename DataType>
using GatherIndices = GatherFixture<DataType, IndexDataType>;
TYPED_TEST_SUITE(GatherIndices, GTestTypeList);

TYPED_TEST(GatherIndices, G2D_Axis0_NegIndice) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1., 2., 3., 4.};
  const std::vector<IndexDataType> indices = {-5};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}

TYPED_TEST(GatherIndices, G2D_Axis0_InvIndice) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {0, 0, 0, 0};
  const std::vector<IndexDataType> indices = {100};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {1};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}

TYPED_TEST(GatherIndices, G2D_Axis0_MixedNegIndice) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {5, 6, 7, 8, 5, 6, 7, 8};
  const std::vector<IndexDataType> indices = {-4, 1};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {2};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}

TYPED_TEST(GatherIndices, G2D_Axis0_MixedInvIndice) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {0, 0, 0, 0, 5, 6, 7, 8};
  const std::vector<IndexDataType> indices = {-100, 1};
  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {2};
  params.input_dims = {5, 4};
  this->test_gather(exp_out, params, indices);
}
