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
#ifndef SYCLDNN_TEST_ND_REDUCE_FIXTURE_H_
#define SYCLDNN_TEST_ND_REDUCE_FIXTURE_H_

#include <gtest/gtest.h>
#include <vector>

#include "sycldnn/helpers/scope_exit.h"
#include "sycldnn/reduce/helpers.h"
#include "sycldnn/reduce/launch.h"
#include "sycldnn/reduce/params.h"
#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

template <typename Pair, typename Op>
struct NdReduceFixture : public BackendTestFixture<typename Pair::SecondType> {
  using DataType = typename Pair::FirstType;
  using Backend = typename Pair::SecondType;

 protected:
  void run(std::vector<DataType> const& exp, std::vector<int> dims,
           std::vector<int> axes, DataType max_val) {
    // Sort so the last element in the axes is the inner most dimmension
    std::sort(axes.begin(), axes.end());
    sycldnn::reduce::NDReduceParams params;
    params.dims = dims;
    params.axes = axes;
    params.prepare_output = true;
    auto const sizes = sycldnn::reduce::get_sizes(params);

    ASSERT_EQ(sizes.output_size, exp.size());

    std::vector<DataType> input_data =
        iota_initialised_data(sizes.input_size, max_val);
    std::vector<DataType> output_data(sizes.output_size, 0);
    iota_initialised_data(sizes.output_size, max_val);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();
    {
      auto input_gpu =
          provider.get_initialised_device_memory(sizes.input_size, input_data);
      auto output_gpu = provider.get_initialised_device_memory(
          sizes.output_size, output_data);
      auto workspace_size = sycldnn::reduce::get_workspace_size(params);
      auto workspace =
          provider.get_backend().template allocate<DataType>(workspace_size);

      SNN_ON_SCOPE_EXIT {
        provider.deallocate_ptr(input_gpu);
        provider.deallocate_ptr(output_gpu);
        provider.deallocate_ptr(workspace);
      };
      provider.get_backend().get_queue().wait();

      auto status = sycldnn::reduce::launch<DataType, Op>(
          input_gpu, output_gpu, params, backend, workspace, workspace_size);

      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
      provider.get_backend().get_queue().wait_and_throw();

      provider.copy_device_data_to_host(sizes.output_size, output_gpu,
                                        output_data);
    }

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(exp[i], output_data[i], 10u);
    }
  }
};

#endif  // SYCLDNN_TEST_ND_REDUCE_FIXTURE_H_
