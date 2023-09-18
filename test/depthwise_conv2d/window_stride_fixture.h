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
#ifndef PORTDNN_TEST_DEPTHWISE_CONV2D_WINDOW_STIRDE_FIXTURE_H_
#define PORTDNN_TEST_DEPTHWISE_CONV2D_WINDOW_STIRDE_FIXTURE_H_
#include <gtest/gtest.h>

#include "portdnn/depthwise_conv2d/params.h"

#include "portdnn/helpers/padding.h"

#include "test/depthwise_conv2d/depthwise_conv2d_fixture.h"

#include <array>
#include <vector>

namespace sycldnn {
namespace depthwise_conv2d {

template <typename Pair, int Window, int Stride>
struct WindowStrideTest : public DepthwiseConv2DFixture<Pair> {
  using typename DepthwiseConv2DFixture<Pair>::DataType;

 protected:
  void run_forward_test(std::vector<DataType> const& exp_out,
                        std::array<int, 4> const& in_shape, int multiplier,
                        sycldnn::PaddingMode padding, DataType max_val) {
    auto params = get_params(in_shape, multiplier, padding);
    test_forward(exp_out, params, max_val);
  }
  void run_input_backprop_test(std::vector<DataType> const& exp_out,
                               std::array<int, 4> const& in_shape,
                               int multiplier, sycldnn::PaddingMode padding,
                               DataType max_val) {
    auto params = get_params(in_shape, multiplier, padding);
    test_input_backprop(exp_out, params, max_val);
  }
  void run_filter_backprop_test(std::vector<DataType> const& exp_out,
                                std::array<int, 4> const& in_shape,
                                int multiplier, sycldnn::PaddingMode padding,
                                DataType max_val) {
    auto params = get_params(in_shape, multiplier, padding);
    test_filter_backprop(exp_out, params, max_val);
  }

 private:
  DepthwiseConv2DParams get_params(std::array<int, 4> const& in_shape,
                                   int multiplier,
                                   sycldnn::PaddingMode padding) {
    DepthwiseConv2DParams params{};
    params.channels = in_shape[3];
    params.channel_multiplier = multiplier;
    params.batch = in_shape[0];
    params.in_rows = in_shape[1];
    params.in_cols = in_shape[2];
    params.window_rows = Window;
    params.window_cols = Window;
    params.stride_rows = Stride;
    params.stride_cols = Stride;
    return sycldnn::helpers::add_padding_to(params, padding);
  }
  void test_forward(std::vector<DataType> const& exp,
                    DepthwiseConv2DParams const& params, DataType max_val) {
    using sycldnn::conv2d::conv_type::Forward;
    SCOPED_TRACE("Forward pass");
    this->template test_conv<Forward>(exp, params, max_val);
  }
  void test_input_backprop(std::vector<DataType> const& exp,
                           DepthwiseConv2DParams const& params,
                           DataType max_val) {
    using sycldnn::conv2d::conv_type::InputBackprop;
    SCOPED_TRACE("Input backprop pass");
    this->template test_conv<InputBackprop>(exp, params, max_val);
  }
  void test_filter_backprop(std::vector<DataType> const& exp,
                            DepthwiseConv2DParams const& params,
                            DataType max_val) {
    using sycldnn::conv2d::conv_type::FilterBackprop;
    SCOPED_TRACE("Filter backprop pass");
    this->template test_conv<FilterBackprop>(exp, params, max_val);
  }
};

}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_TEST_DEPTHWISE_CONV2D_WINDOW_STIRDE_FIXTURE_H_
