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
#include "test/types/kernel_data_types.h"

#include "test/pooling/pooling_event_dependencies_fixture.h"
#include "test/pooling/pooling_fixture.h"

using GTestTypeList = sycldnn::types::GTestKernelDataTypes;
template <typename DType>
using TestFixture =
    PoolingEventDependenciesFixture<DType, sycldnn::pooling::Max,
                                    sycldnn::pooling::Forward>;

TYPED_TEST_SUITE(TestFixture, GTestTypeList);
TYPED_TEST(TestFixture, EVENT_DEPENDENCIES) {
  const std::array<int, 4> in_shape = {{1, 4, 4, 2}};
  const auto padding = sycldnn::PaddingMode::SAME;
  const auto params = getPoolingParams<3, 1>(in_shape, padding);
  this->test_pool_event_dependencies(params);
}
