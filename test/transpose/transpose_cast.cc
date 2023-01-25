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

#include "test/transpose/transpose_fixture.h"
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

using Backends = sycldnn::types::DefaultBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

template <typename Pair>
using TransposeCast = TransposeFixture<Pair>;
TYPED_TEST_SUITE(TransposeCast, GTestTypePairs);

TYPED_TEST(TransposeCast, T3D_2x3x4_0x2x1) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {1,  5,  9,  2,  6,  10, 3,  7,
                                         11, 4,  8,  12, 13, 17, 21, 14,
                                         18, 22, 15, 19, 23, 16, 20, 24};
  const std::vector<int> sizes = {2, 3, 4};
  const std::vector<int> perm = {0, 2, 1};
  const DataType max_input_val = 127;
  this->run(exp_out, sizes, perm, max_input_val, 0, 0);
}
