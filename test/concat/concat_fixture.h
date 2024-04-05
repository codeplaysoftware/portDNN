/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#ifndef SYCLDNN_TEST_CONCAT_CONCAT_FIXTURE_H_
#define SYCLDNN_TEST_CONCAT_CONCAT_FIXTURE_H_

#include <gtest/gtest.h>

#include "sycldnn/concat/launch.h"
#include "sycldnn/concat/params.h"
#include "sycldnn/concat/tarray.h"

#include "sycldnn/helpers/scope_exit.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"

#include <vector>

template <typename T, typename Backend>
struct ConcatFixture : public BackendTestFixture<Backend> {
  using DataType = T;

  void test_concat(std::vector<T> exp, std::vector<int> input_sizes,
                   sycldnn::concat::ConcatParams& params) {
    std::vector<std::vector<T>> inputs;

    int running_total = 0;
    for (auto s : input_sizes) {
      std::vector<T> tensor;
      internal::iota_n<T>(tensor, s, running_total);
      inputs.push_back(tensor);
      running_total += s;
    }
    auto out_size = sycldnn::concat::internal::get_concat_output_size(params);
    std::vector<T> output(out_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    std::vector<typename Backend::template pointer_type<const T>> input_vec_gpu;
    for (auto& inp : inputs) {
      input_vec_gpu.push_back(
          provider.get_initialised_device_memory(inp.size(), inp));
    }

    auto out_gpu = provider.get_initialised_device_memory(out_size, output);

    auto status = sycldnn::concat::launch<T, Backend>(input_vec_gpu, out_gpu,
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

#endif
