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
#ifndef PORTDNN_TEST_DEPTHWISE_CONV2D_DEPTHWISE_CONV2D_FIXTURE_H_
#define PORTDNN_TEST_DEPTHWISE_CONV2D_DEPTHWISE_CONV2D_FIXTURE_H_

#include <gtest/gtest.h>
#include <vector>

#include "portdnn/depthwise_conv2d/launch.h"
#include "portdnn/depthwise_conv2d/params.h"
#include "portdnn/depthwise_conv2d/sizes.h"

#include "portdnn/helpers/scope_exit.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"
#include "test/helpers/float_comparison.h"

namespace sycldnn {
namespace depthwise_conv2d {

template <typename T>
struct DepthwiseConv2DEventFixture
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = T;

 protected:
  /** Test a convolution with both input and filter set to `1, 2, 3,...` */
  template <typename ConvType>
  void test_conv(sycldnn::depthwise_conv2d::DepthwiseConv2DParams const& params,
                 DataType max_val = static_cast<DataType>(0)) {
    auto conv_sizes = sycldnn::depthwise_conv2d::get_sizes<ConvType>(params);

    std::vector<DataType> input =
        iota_initialised_data(conv_sizes.input_size, max_val);
    std::vector<DataType> filter =
        iota_initialised_data(conv_sizes.filter_size, max_val);
    std::vector<DataType> output(conv_sizes.output_size,
                                 static_cast<DataType>(0));

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu =
        provider.get_initialised_device_memory(conv_sizes.input_size, input);
    auto fil_gpu =
        provider.get_initialised_device_memory(conv_sizes.filter_size, filter);
    auto out_gpu =
        provider.get_initialised_device_memory(conv_sizes.output_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(fil_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    dependency_test_params dep_test_params;
    cl::sycl::event dependee_e = create_event(backend, dep_test_params);

    auto status = sycldnn::depthwise_conv2d::launch<DataType, ConvType>(
        inp_gpu, fil_gpu, out_gpu, params, backend, {dependee_e});

    if (status.status == sycldnn::StatusCode::InvalidAlgorithm) {
      // Do not check results if the implementation is not supported.
      GTEST_SKIP() << "Skipping test because implementation is not supported.";
    }
    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);

    check_dependency(dependee_e, status.event, backend, dep_test_params);
  }
};

}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_TEST_DEPTHWISE_CONV2D_DEPTHWISE_CONV2D_FIXTURE_H_
