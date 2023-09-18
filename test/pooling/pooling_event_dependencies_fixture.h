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

#ifndef PORTDNN_TEST_POOLING_POOLING_EVENT_DEPENDENCIES_FIXTURE_H_
#define PORTDNN_TEST_POOLING_POOLING_EVENT_DEPENDENCIES_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/padding_mode.h"

#include "portdnn/helpers/padding.h"
#include "portdnn/helpers/scope_exit.h"

#include "portdnn/pooling/launch.h"
#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"
#include "portdnn/pooling/sizes.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"

#include <vector>

template <typename DType, template <typename T> class Op, typename Direction>
struct PoolingEventDependenciesFixture
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = DType;

  void test_pool_event_dependencies(sycldnn::pooling::PoolingParams params,
                                    DataType max_val = DataType{0},
                                    size_t in_offset = 0u,
                                    size_t out_offset = 0u) {
    auto pooling_size = sycldnn::pooling::get_sizes<Direction>(params);
    auto in_size = pooling_size.input_size;
    auto out_size = pooling_size.output_size + out_offset;
    std::vector<DataType> input = iota_initialised_data(in_size, max_val);
    input.insert(input.begin(), in_offset, 0);
    std::vector<DataType> output(out_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input.size(), input);
    auto out_gpu = provider.get_initialised_device_memory(out_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    dependency_test_params dep_test_params;
    cl::sycl::event dependee_e = create_event(backend, dep_test_params);

    auto status = sycldnn::pooling::launch<DataType, Op, Direction>(
        inp_gpu + in_offset, out_gpu + out_offset, params, backend,
        {dependee_e});

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    check_dependency(dependee_e, status.event, backend, dep_test_params);
  }
};

#endif  // PORTDNN_TEST_POOLING_POOLING_EVENT_DEPENDENCIES_FIXTURE_H_
