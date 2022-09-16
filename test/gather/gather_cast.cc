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

using IntDataTypeList =
    sycldnn::types::TypeList<int8_t, int16_t, int32_t, int64_t, uint8_t,
                             uint16_t, uint32_t, uint64_t>;
using DataTypeList =
    sycldnn::types::Concatenate<sycldnn::types::KernelDataTypes,
                                IntDataTypeList>::type;
using GTestTypeList = sycldnn::types::ToGTestTypes<DataTypeList>::type;

using IndexDataType = int32_t;  // or int64_t

template <typename DataType>
using GatherCastTestFixture = GatherFixture<DataType, IndexDataType>;
TYPED_TEST_SUITE(GatherCastTestFixture, GTestTypeList);

TYPED_TEST(GatherCastTestFixture, INDEX_1_AXIS_0_INPUT_2) {
  using DataType = typename TestFixture::DataType;

  const std::vector<DataType> exp_out = {5, 6, 7, 8};
  const std::vector<IndexDataType> indices = {1};

  gather::GatherParams params;
  params.axis = 0;
  params.indices_dims = {static_cast<int>(indices.size())};
  params.input_dims = {3, 4};

  this->test_gather(exp_out, params, indices);
}
