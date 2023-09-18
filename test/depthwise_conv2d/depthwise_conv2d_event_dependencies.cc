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

#include "test/depthwise_conv2d/depthwise_conv2d_event_dependencies_fixture.h"

#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>

template <typename Pair>
using OneByOneInputDepthConvTest =
    sycldnn::depthwise_conv2d::DepthwiseConv2DEventFixture<Pair>;

using DataTypeList = sycldnn::types::KernelDataTypes;

using GTestDataTypeList = sycldnn::types::ToGTestTypes<DataTypeList>::type;
TYPED_TEST_SUITE(OneByOneInputDepthConvTest, GTestDataTypeList);

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

TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1_FR) {
  auto params = get_1x1_params(1, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1_FR) {
  auto params = get_1x1_params(1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1_FR) {
  auto params = get_1x1_params(2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep2Features1x1_FR) {
  auto params = get_1x1_params(2, 3, 2);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1And3x3Filter_FR) {
  auto params = get_1x1_params_fxf_filter(1, 1, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1And3x3Filter_FR) {
  auto params = get_1x1_params_fxf_filter(1, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter_FR) {
  auto params = get_1x1_params_fxf_filter(2, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter2Features_FR) {
  auto params = get_1x1_params_fxf_filter(2, 3, 2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(params);
}

TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1_FB) {
  auto params = get_1x1_params(1, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1_FB) {
  auto params = get_1x1_params(1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1_FB) {
  auto params = get_1x1_params(2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep2Features1x1_FB) {
  auto params = get_1x1_params(2, 3, 2);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1And3x3Filter_FB) {
  auto params = get_1x1_params_fxf_filter(1, 1, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1And3x3Filter_FB) {
  auto params = get_1x1_params_fxf_filter(1, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter_FB) {
  auto params = get_1x1_params_fxf_filter(2, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter2Features_FB) {
  auto params = get_1x1_params_fxf_filter(2, 3, 2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(params);
}

TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1_IB) {
  auto params = get_1x1_params(1, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1_IB) {
  auto params = get_1x1_params(1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1_IB) {
  auto params = get_1x1_params(2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep2Features1x1_IB) {
  auto params = get_1x1_params(2, 3, 2);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1And3x3Filter_IB) {
  auto params = get_1x1_params_fxf_filter(1, 1, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1And3x3Filter_IB) {
  auto params = get_1x1_params_fxf_filter(1, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter_IB) {
  auto params = get_1x1_params_fxf_filter(2, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter2Features_IB) {
  auto params = get_1x1_params_fxf_filter(2, 3, 2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(params);
}
