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
#ifndef PORTDNN_TEST_CONV2D_WINDOW_STRIDE_FIXTURE_H_
#define PORTDNN_TEST_CONV2D_WINDOW_STRIDE_FIXTURE_H_
#include <gtest/gtest.h>

#include "portdnn/conv2d/params.h"

#include "portdnn/helpers/padding.h"

#include "test/conv2d/convolution_fixture.h"

#include <vector>

template <typename Tuple, int Window, int Stride>
struct WindowStrideTest : public ConvolutionFixture<Tuple> {
  using typename ConvolutionFixture<Tuple>::DataType;
  using Conv2DParams = sycldnn::conv2d::Conv2DParams;

 protected:
  void run_forward_test(std::vector<DataType> const& exp_out,
                        std::array<int, 4> const& in_shape, int features,
                        sycldnn::PaddingMode padding, DataType max_val) {
    auto params = get_params(in_shape, features, padding);
    test_forward(exp_out, params, max_val);
  }
  void run_input_backprop_test(std::vector<DataType> const& exp_out,
                               std::array<int, 4> const& in_shape, int features,
                               sycldnn::PaddingMode padding, DataType max_val) {
    auto params = get_params(in_shape, features, padding);
    test_input_backprop(exp_out, params, max_val);
  }
  void run_filter_backprop_test(std::vector<DataType> const& exp_out,
                                std::array<int, 4> const& in_shape,
                                int features, sycldnn::PaddingMode padding,
                                DataType max_val) {
    auto params = get_params(in_shape, features, padding);
    test_filter_backprop(exp_out, params, max_val);
  }

 private:
  Conv2DParams get_params(std::array<int, 4> const& in_shape, int features,
                          sycldnn::PaddingMode padding) {
    sycldnn::conv2d::Conv2DParams params{};
    params.channels = in_shape[3];
    params.features = features;
    params.batch = in_shape[0];
    params.in_rows = in_shape[1];
    params.in_cols = in_shape[2];
    params.window_rows = Window;
    params.window_cols = Window;
    params.stride_rows = Stride;
    params.stride_cols = Stride;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return sycldnn::helpers::add_padding_to(params, padding);
  }
  void test_forward(std::vector<DataType> const& exp,
                    Conv2DParams const& params, DataType max_val) {
    using sycldnn::conv2d::conv_type::Forward;
    SCOPED_TRACE("Forward pass");
    this->template test_conv<Forward>(exp, params, max_val);
  }
  void test_input_backprop(std::vector<DataType> const& exp,
                           Conv2DParams const& params, DataType max_val) {
    using sycldnn::conv2d::conv_type::InputBackprop;
    SCOPED_TRACE("Input backprop pass");
    this->template test_conv<InputBackprop>(exp, params, max_val);
  }
  void test_filter_backprop(std::vector<DataType> const& exp,
                            Conv2DParams const& params, DataType max_val) {
    using sycldnn::conv2d::conv_type::FilterBackprop;
    SCOPED_TRACE("Filter backprop pass");
    this->template test_conv<FilterBackprop>(exp, params, max_val);
  }
};
#endif  // PORTDNN_TEST_CONV2D_WINDOW_STRIDE_FIXTURE_H_
