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

#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/sizes.h"

#include "src/backend/snn_backend_provider.h"
#include "src/backend/snn_usm_backend_provider.h"
#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"
#include "test/backend/backend_test_fixture.h"

#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <memory>

#include <CL/sycl.hpp>

template <typename Backend>
struct DefaultSelectorFixture : public BackendTestFixture<Backend> {
 protected:
  void check_conv_launch_successful(
      sycldnn::conv2d::Conv2DParams const& params) {
    using ConvType = sycldnn::conv2d::conv_type::Forward;

    using HostData = std::vector<float>;

    auto& provider = this->provider_;
    auto device = provider.get_backend().get_queue().get_device();

    auto sizes = sycldnn::conv2d::get_sizes<ConvType>(params);
    HostData input(sizes.input_size);
    HostData filter(sizes.filter_size);
    HostData output(sizes.output_size);

    auto input_gpu =
        provider.get_initialised_device_memory(sizes.input_size, input);
    auto filter_gpu =
        provider.get_initialised_device_memory(sizes.filter_size, filter);
    auto output_gpu =
        provider.get_initialised_device_memory(sizes.output_size, output);

    // Use the queue to get a device reference, then validate that we return a
    // non-null selector.
    auto selector = sycldnn::conv2d::get_default_selector(device);
    EXPECT_TRUE(nullptr != selector);
    auto status = sycldnn::conv2d::launch<float, ConvType>(
        input_gpu, filter_gpu, output_gpu, params, *selector,
        provider.get_backend());
    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();
  }
};

template <typename Backend>
using DefaultSelectorTest = DefaultSelectorFixture<Backend>;

TYPED_TEST_SUITE(DefaultSelectorTest, sycldnn::types::GTestDefaultBackendTypes);

TYPED_TEST(DefaultSelectorTest, GetValidSelectionFor5x5s2) {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 3;
  params.features = 32;
  params.batch = 5;
  params.in_rows = 128;
  params.in_cols = 128;
  params.window_rows = 5;
  params.window_cols = 5;
  params.stride_rows = 2;
  params.stride_cols = 2;
  params.out_rows = 64;
  params.out_cols = 64;
  params.pad_rows = 1;
  params.pad_cols = 1;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  this->check_conv_launch_successful(params);
}
