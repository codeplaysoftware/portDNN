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

#ifndef PORTDNN_TEST_BIAS_BIAS_FIXTURE_H_
#define PORTDNN_TEST_BIAS_BIAS_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/helpers/dims.h"
#include "portdnn/helpers/padding.h"
#include "portdnn/helpers/scope_exit.h"

#include "portdnn/binaryop/launch.h"
#include "portdnn/binaryop/operators.h"
#include "portdnn/binaryop/params.h"
#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

#include <string>
#include <vector>

template <typename DType, typename Backend>
struct BiasFixture : public BackendTestFixture<Backend> {
  using DataType = DType;

  void test_bias(std::vector<DataType> exp,
                 sycldnn::binaryop::BinaryParams const& params,
                 DataType max_val = DataType{0}) {
    auto in_size = sycldnn::helpers::get_total_size(params.lhs_dims);
    auto out_size = sycldnn::helpers::get_total_size(params.lhs_dims);
    auto bias_size = sycldnn::helpers::get_total_size(params.rhs_dims);
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

    auto status = sycldnn::binaryop::launch<DataType, sycldnn::binaryop::Add>(
        inp_gpu, bias_gpu, out_gpu, params, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(out_size, out_gpu, output);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      EXPECT_EQ(exp[i], output[i]);
    }
  }
};

inline sycldnn::binaryop::BinaryParams getBiasParams(
    std::array<int, 4> in_shape) {
  sycldnn::binaryop::BinaryParams ret;
  int tot_size = sycldnn::helpers::get_total_size(in_shape);
  ret.lhs_dims = {tot_size / in_shape[3], in_shape[3]};
  ret.rhs_dims = {1, in_shape[3]};
  return ret;
}

#endif  // PORTDNN_TEST_BIAS_BIAS_FIXTURE_H_
