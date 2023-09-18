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

#ifndef PORTDNN_TEST_GATHER_GATHER_FIXTURE_H_
#define PORTDNN_TEST_GATHER_GATHER_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/gather/launch.h"
#include "portdnn/gather/sizes.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"

#include <numeric>
#include <string>
#include <vector>

template <typename Pair, typename Index>
struct GatherFixture : public BackendTestFixture<typename Pair::SecondType> {
  using DataType = typename Pair::FirstType;

  void test_gather(std::vector<DataType> const& exp,
                   sycldnn::gather::GatherParams const& params,
                   std::vector<Index> const& indices,
                   DataType max_val = DataType{0}) {
    sycldnn::gather::GatherSizes gather_sizes =
        sycldnn::gather::get_sizes(params);

    auto in_size = gather_sizes.input_size;
    auto indices_size = gather_sizes.indices_size;
    auto out_size = gather_sizes.output_size;

    std::vector<DataType> input = iota_initialised_data(in_size, max_val);
    std::vector<DataType> output(out_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(in_size, input);
    auto indices_gpu =
        provider.get_initialised_device_memory(indices_size, indices);
    auto out_gpu = provider.get_initialised_device_memory(out_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(indices_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::gather::launch<DataType, Index>(
        inp_gpu, indices_gpu, out_gpu, params, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(out_size, out_gpu, output);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      EXPECT_EQ(exp[i], output[i]);
    }
  }
};

#endif  // PORTDNN_TEST_GATHER_GATHER__FIXTURE_H_
