/*
 * Copyright 2018 Codeplay Software Ltd.
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

#ifndef SYCLDNN_TEST_POOLING_POOLING_FIXTURE_H_
#define SYCLDNN_TEST_POOLING_POOLING_FIXTURE_H_

#include <gtest/gtest.h>

#include "sycldnn/padding_mode.h"

#include "sycldnn/helpers/padding.h"
#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

#include <string>
#include <vector>

template <typename DType, typename Backend, template <typename T> class Op,
          typename Direction, bool = sycldnn::pooling::internal::IsMaxGradient<
                                  DType, Op, Direction>::value>
struct PoolingFixture : public BackendTestFixture<Backend> {
  using DataType = DType;

  void test_pool(std::vector<DataType> exp,
                 sycldnn::pooling::PoolingParams const& params,
                 DataType max_val = DataType{0}) {
    auto pooling_size = sycldnn::pooling::get_sizes<Direction>(params);
    auto in_size = pooling_size.input_size;
    auto out_size = pooling_size.output_size;
    std::vector<DataType> input = iota_initialised_data(in_size, max_val);
    std::vector<DataType> output(out_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(in_size, input);
    auto out_gpu = provider.get_initialised_device_memory(out_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::pooling::launch<DataType, Op, Direction>(
        inp_gpu, out_gpu, params, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(out_size, out_gpu, output);

    auto platform = backend.get_queue().get_device().get_platform();
    auto plat_profile =
        platform.template get_info<cl::sycl::info::platform::profile>();
    size_t tolerance = 0u;

    // Taking the average pooling gradient can result in significantly higher
    // error in the results than for other pooling ops. Allow for a greater
    // margin of error when doing average gradient.
    bool is_avg_grad =
        sycldnn::pooling::internal::IsAverageGradient<DataType, Op,
                                                      Direction>::value;

    if (plat_profile.find("FULL_PROFILE") != std::string::npos) {
      tolerance = is_avg_grad ? 8u : 4u;
    } else if (plat_profile.find("EMBEDDED_PROFILE") != std::string::npos) {
      tolerance = is_avg_grad ? 48u : 4u;
    }

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(exp[i], output[i], tolerance);
    }
  }
};

// Need a specific fixture for maxpooling gradient, as this operation requires
// both the original pooling values and the backprop values.
template <typename DType, typename Backend, template <typename> class Op,
          typename Direction>
struct PoolingFixture<DType, Backend, Op, Direction, true>
    : public BackendTestFixture<Backend> {
  using DataType = DType;

  void test_pool(std::vector<DataType> exp,
                 sycldnn::pooling::PoolingParams const& params,
                 DataType max_val = DataType{0}) {
    auto pooling_size =
        sycldnn::pooling::get_sizes<sycldnn::pooling::Forward>(params);
    auto in_size = pooling_size.input_size;
    auto out_size = pooling_size.output_size;

    std::vector<DataType> input_data = iota_initialised_data(in_size, max_val);
    std::vector<DataType> output_data(out_size);
    std::vector<DataType> input_backprop =
        iota_initialised_data(out_size, max_val);
    std::vector<DataType> output_backprop(in_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_data_gpu =
        provider.get_initialised_device_memory(in_size, input_data);
    auto out_data_gpu =
        provider.get_initialised_device_memory(out_size, output_data);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_data_gpu);
      provider.deallocate_ptr(out_data_gpu);
    };

    auto fwd_status = sycldnn::pooling::launch<DataType, sycldnn::pooling::Max,
                                               sycldnn::pooling::Forward>(
        inp_data_gpu, out_data_gpu, params, backend);
    ASSERT_EQ(sycldnn::StatusCode::OK, fwd_status.status);

    auto inp_backprop_gpu =
        provider.get_initialised_device_memory(out_size, input_backprop);
    auto out_backprop_gpu =
        provider.get_initialised_device_memory(in_size, output_backprop);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_backprop_gpu);
      provider.deallocate_ptr(out_backprop_gpu);
    };

    auto back_status =
        sycldnn::pooling::launch<DataType, sycldnn::pooling::Max,
                                 sycldnn::pooling::Backpropagate>(
            inp_data_gpu, out_data_gpu, inp_backprop_gpu, out_backprop_gpu,
            params, backend);
    ASSERT_EQ(sycldnn::StatusCode::OK, back_status.status);

    fwd_status.event.wait_and_throw();
    back_status.event.wait_and_throw();

    provider.copy_device_data_to_host(in_size, out_backprop_gpu,
                                      output_backprop);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(exp[i], output_backprop[i], 0u);
    }
  }
};

template <int Window, int Stride>
inline sycldnn::pooling::PoolingParams getPoolingParams(
    std::array<int, 4> in_shape, sycldnn::PaddingMode pad) {
  sycldnn::pooling::PoolingParams ret;
  ret.in_rows = in_shape[1];
  ret.in_cols = in_shape[2];
  ret.window_rows = Window;
  ret.window_cols = Window;
  ret.stride_rows = Stride;
  ret.stride_cols = Stride;
  ret.batch = in_shape[0];
  ret.channels = in_shape[3];

  ret = sycldnn::helpers::add_padding_to(ret, pad);
  return ret;
}

#endif  // SYCLDNN_TEST_POOLING_POOLING_FIXTURE_H_
