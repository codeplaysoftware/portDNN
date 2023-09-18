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
#ifndef PORTDNN_TEST_TRANSPOSE_TRANSPOSE_FIXTURE_H_
#define PORTDNN_TEST_TRANSPOSE_TRANSPOSE_FIXTURE_H_

#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/transpose/launch.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

template <typename Pair>
struct TransposeFixture : public BackendTestFixture<typename Pair::SecondType> {
  using DataType = typename Pair::FirstType;

 protected:
  void run(std::vector<DataType> const& exp, std::vector<int> const& sizes,
           std::vector<int> const& permutation, DataType max_val,
           size_t in_offset, size_t out_offset) {
    size_t tensor_size = std::accumulate(begin(sizes), end(sizes), 1,
                                         [](int a, int b) { return a * b; });
    size_t in_size = tensor_size + in_offset;
    size_t out_size = tensor_size + out_offset;
    ASSERT_EQ(out_size, exp.size());

    std::vector<DataType> in_data = iota_initialised_data(in_size, max_val);
    std::vector<DataType> out_data = iota_initialised_data(out_size, max_val);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    {
      auto in_gpu = provider.get_initialised_device_memory(in_size, in_data);
      auto out_gpu = provider.get_initialised_device_memory(out_size, out_data);

      SNN_ON_SCOPE_EXIT {
        provider.deallocate_ptr(in_gpu);
        provider.deallocate_ptr(out_gpu);
      };

      try {
        auto status = sycldnn::transpose::launch<DataType>(
            in_gpu + in_offset, out_gpu + out_offset, sizes, permutation,
            backend);

        ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        throw std::runtime_error(e.what());
      }

      provider.copy_device_data_to_host(out_size, out_gpu, out_data);
    }

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      EXPECT_EQ(exp[i], out_data[i]);
    }
  }
};
#endif  // PORTDNN_TEST_TRANSPOSE_TRANSPOSE_FIXTURE_H_
