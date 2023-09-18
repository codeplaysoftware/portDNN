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

#include "portdnn/conv2d/conv_type.h"

#include "portdnn/depthwise_conv2d/params.h"

#include "test/depthwise_conv2d/depthwise_conv2d_fixture.h"

#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>

template <typename Pair>
using OneByOneInputDepthConvTest =
    sycldnn::depthwise_conv2d::DepthwiseConv2DFixture<Pair>;

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using BackendTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;
using GTestTypeTriples = sycldnn::types::ToGTestTypes<BackendTypePairs>::type;
TYPED_TEST_SUITE(OneByOneInputDepthConvTest, GTestTypeTriples);

sycldnn::depthwise_conv2d::DepthwiseConv2DParams get_1x1_params(
    int batch, int channels, int chan_mult = 1) {
  sycldnn::depthwise_conv2d::DepthwiseConv2DParams params;
  params.channels = channels;
  params.channel_multiplier = chan_mult;
  params.batch = batch;
  params.in_rows = 1;
  params.in_cols = 1;
  params.window_rows = 1;
  params.window_cols = 1;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 1;
  params.out_cols = 1;
  params.pad_rows = 0;
  params.pad_cols = 0;
  return params;
}

sycldnn::depthwise_conv2d::DepthwiseConv2DParams get_1x1_params_fxf_filter(
    int batch, int channels, int chan_mult, int window_size) {
  auto params = get_1x1_params(batch, channels, chan_mult);
  params.window_rows = window_size;
  params.window_cols = window_size;
  params.pad_rows = (window_size - 1) / 2;
  params.pad_cols = (window_size - 1) / 2;
  return params;
}

/**
 * Input: 1     Out deltas: 1
 *
 * Filter deltas: 1
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1};
  auto params = get_1x1_params(1, 1);
  const DataType max_input_val = 1.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
/**
 * Input: 1     Out deltas: 1
 *         2                 2
 *          3                 3
 *
 * Filter deltas: 1
 *                 4
 *                  9
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1, 4, 9};
  auto params = get_1x1_params(1, 3);
  const DataType max_input_val = 3.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
/**
 * Input: 1   4    Out deltas: 1   4
 *         2   5                2   5
 *          3   6                3   6
 *
 * Filter deltas: 1+16
 *                 4+25
 *                  9+36
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {17, 29, 45};
  auto params = get_1x1_params(2, 3);
  const DataType max_input_val = 6.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
/**
 * Input: 1   4    Out deltas: 1   7
 *         2   5                2   8
 *          3   6                3   9
 *                                4   10
 *                                 5   11
 *                                  6   12
 *
 * Filter deltas: 1+28
 *                 2+32
 *                  6+45
 *                   8+50
 *                    15+66
 *                     18+72
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep2Features1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {29, 34, 51, 58, 81, 90};
  auto params = get_1x1_params(2, 3, 2);
  const DataType max_input_val = 12.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
/**
 * Input:  1    Out deltas: 1
r*
 * Filter deltas: 0 0 0
 *                0 1 0
 *                0 0 0
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  auto params = get_1x1_params_fxf_filter(1, 1, 1, 3);
  const DataType max_input_val = 1.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
/**
 * Input:            Out deltas:
 *          1                     1
 *           2                     2
 *            3                     3
 *
 * Filter deltas: 0 0 0
 *                0 1 0
 *                0 0 0
 *
 *                  0 0 0
 *                  0 4 0
 *                  0 0 0
 *
 *                    0 0 0
 *                    0 9 0
 *                    0 0 0
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4,
                               9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto params = get_1x1_params_fxf_filter(1, 3, 1, 3);
  const DataType max_input_val = 3.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
/**
 * Input:            Out deltas:
 *          1   4                 1   4
 *           2   5                 2   5
 *            3   6                 3   6
 *
 * Filter deltas: 0  0   0
 *                0 1+16 0
 *                0  0   0
 *
 *                  0  0   0
 *                  0 4+25 0
 *                  0  0   0
 *
 *                    0  0   0
 *                    0 9+36 0
 *                    0  0   0
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 29,
                               45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto params = get_1x1_params_fxf_filter(2, 3, 1, 3);
  const DataType max_input_val = 6.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
/**
 * Input:            Out deltas:
 *          1   4                 1   7
 *           2   5                 2   8
 *            3   6                 3   9
 *                                   4   10
 *                                    5   11
 *                                     6   12
 *
 * Filter deltas: 0  0   0
 *                0 1+28 0
 *                0  0   0
 *
 *                  0  0   0
 *                  0 2+32 0
 *                  0  0   0
 *
 *                     0  0   0
 *                     0 6+45 0
 *                     0  0   0
 *
 *                        0  0   0
 *                        0 8+50 0
 *                        0  0   0
 *
 *                           0   0   0
 *                           0 15+66 0
 *                           0   0   0
 *
 *                              0   0   0
 *                              0 18+72 0
 *                              0   0   0
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter2Features) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,
                               0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 29, 34, 51, 58,
                               81, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,
                               0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0};
  auto params = get_1x1_params_fxf_filter(2, 3, 2, 3);
  const DataType max_input_val = 12.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, max_input_val);
}
