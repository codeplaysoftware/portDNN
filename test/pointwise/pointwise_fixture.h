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

#ifndef PORTDNN_TEST_POINTWISE_POINTWISE_FIXTURE_H_
#define PORTDNN_TEST_POINTWISE_POINTWISE_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/launch.h"
#include "portdnn/pointwise/operators.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

#include <vector>

template <typename Pair, template <typename> class Op, typename Direction>
struct PointwiseFixture : public BackendTestFixture<typename Pair::SecondType> {
  using DataType = typename Pair::FirstType;

  void test_pointwise(const std::vector<DataType>& input,
                      const std::vector<DataType>& exp) {
    const auto size = exp.size();

    std::vector<DataType> output(size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(size, input);
    auto out_gpu = provider.get_initialised_device_memory(size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::pointwise::launch<DataType, Op, Direction>(
        inp_gpu, out_gpu, size, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(size, out_gpu, output);

    for (size_t i = 0; i < size; ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(exp[i], output[i], 10u);
    }
  }
};

template <typename Pair, template <typename T> class Op>
struct PointwiseFixture<Pair, Op, sycldnn::pointwise::Gradient>
    : public BackendTestFixture<typename Pair::SecondType> {
  using DataType = typename Pair::FirstType;

  void test_pointwise(const std::vector<DataType>& input,
                      const std::vector<DataType>& exp) {
    /* While ULP-based errors are generally better, the expected values
     * in this test are close to 0, which can result in large ULP errors
     * even though the answer is "close" to the expected. */
    const DataType tolerance = 0.00001;
    const auto size = exp.size();

    std::vector<DataType> input_forward = input;
    std::vector<DataType> output_forward(size);
    std::vector<DataType> input_backprop = input;
    std::vector<DataType> output_backprop(size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_fwd_gpu =
        provider.get_initialised_device_memory(size, input_forward);
    auto out_fwd_gpu =
        provider.get_initialised_device_memory(size, output_forward);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_fwd_gpu);
      provider.deallocate_ptr(out_fwd_gpu);
    };

    auto fwd_status =
        sycldnn::pointwise::launch<DataType, Op, sycldnn::pointwise::Forward>(
            inp_fwd_gpu, out_fwd_gpu, size, backend);
    ASSERT_EQ(sycldnn::StatusCode::OK, fwd_status.status);

    auto inp_bk_gpu =
        provider.get_initialised_device_memory(size, input_backprop);
    auto out_bk_gpu =
        provider.get_initialised_device_memory(size, output_backprop);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_bk_gpu);
      provider.deallocate_ptr(out_bk_gpu);
    };

    fwd_status.event.wait_and_throw();

    sycldnn::SNNStatus bk_status;
    if (std::is_same<Op<DataType>, sycldnn::pointwise::Log<DataType>>::value) {
      bk_status = sycldnn::pointwise::launch<DataType, Op,
                                             sycldnn::pointwise::Gradient>(
          inp_fwd_gpu, inp_bk_gpu, out_bk_gpu, size, backend);
    } else {
      bk_status = sycldnn::pointwise::launch<DataType, Op,
                                             sycldnn::pointwise::Gradient>(
          out_fwd_gpu, inp_bk_gpu, out_bk_gpu, size, backend);
    }
    ASSERT_EQ(sycldnn::StatusCode::OK, bk_status.status);

    bk_status.event.wait_and_throw();

    provider.copy_device_data_to_host(size, out_bk_gpu, output_backprop);

    for (size_t i = 0; i < size; ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      EXPECT_NEAR(exp[i], output_backprop[i], tolerance);
    }
  }
};
#endif  // PORTDNN_TEST_POINTWISE_POINTWISE_FIXTURE_H_
