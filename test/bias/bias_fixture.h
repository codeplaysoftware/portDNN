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

#ifndef SYCLDNN_TEST_BIAS_BIAS_FIXTURE_H_
#define SYCLDNN_TEST_BIAS_BIAS_FIXTURE_H_

#include <gtest/gtest.h>

#include "sycldnn/helpers/padding.h"
#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/bias/launch.h"
#include "sycldnn/bias/params.h"
#include "sycldnn/bias/sizes.h"
#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

#include <string>
#include <vector>

template <typename DType, typename Backend>
struct BiasFixture : public BackendTestFixture<Backend> {
  using DataType = DType;

  void test_bias(std::vector<DataType> exp,
                 sycldnn::bias::BiasParams const& params,
                 DataType max_val = DataType{0}) {
    auto size = sycldnn::bias::get_sizes(params);
    auto in_size = size.input_size;
    auto out_size = size.output_size;
    auto bias_size = size.bias_size;
    std::vector<DataType> input = iota_initialised_data(in_size, max_val);
    std::vector<DataType> bias =
        iota_initialised_data(bias_size, static_cast<DataType>(bias_size));
    std::vector<DataType> output(out_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input.size(), input);
    auto bias_gpu = provider.get_initialised_device_memory(bias.size(), bias);
    auto out_gpu = provider.get_initialised_device_memory(out_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(bias_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::bias::launch<DataType>(inp_gpu, bias_gpu, out_gpu,
                                                  params, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(out_size, out_gpu, output);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      EXPECT_EQ(exp[i], output[i]);
    }
  }
};

inline sycldnn::bias::BiasParams getBiasParams(std::array<int, 4> in_shape) {
  sycldnn::bias::BiasParams ret;
  ret.in_rows = in_shape[1];
  ret.in_cols = in_shape[2];
  ret.batch = in_shape[0];
  ret.channels = in_shape[3];
  ret.bias = in_shape[3];
  return ret;
}

#endif  // SYCLDNN_TEST_BIAS_BIAS_FIXTURE_H_
