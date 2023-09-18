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
 * Input: 1     Filter: 1
 *
 * Output: 1
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1};
  auto params = get_1x1_params(1, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/**
 * Input: 1     Filter: 1
 *         2             2
 *          3             3
 *
 * Output: 1 4 9
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1, 4, 9};
  auto params = get_1x1_params(1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/**
 * Input: 1   4    Filter: 1
 *         2   5            2
 *          3   6            3
 *
 * Output: 1 4 9   4  10 18
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1, 4, 9, 4, 10, 18};
  auto params = get_1x1_params(2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/**
 * Input: 1   4    Filter: 1 2
 *         2   5            3 4
 *          3   6            5 6
 *
 * Output: 1  6 15   4  15 30
 *           2  8 18   8  20 36
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep2Features1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1, 2, 6, 8, 15, 18, 4, 8, 15, 20, 30, 36};
  auto params = get_1x1_params(2, 3, 2);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/**
 * Input:       Filter: 1 2 3
 *         1            4 5 6
 *                      7 8 9
 *
 * Output: 5
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Simple1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {5};
  auto params = get_1x1_params_fxf_filter(1, 1, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/**
 * Input:                 Filter: 1  10 19
 *          1                     4  13 22
 *                                7  16 25
 *
 *                                   2  11 20
 *            2                      5  14 23
 *                                   8  17 26
 *
 *                                      3  12 21
 *              3                       6  15 24
 *                                      9  18 27
 *
 *
 * Output: 13 28 45
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, Deep1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {13, 28, 45};
  auto params = get_1x1_params_fxf_filter(1, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/**
 * Input:                  Filter: 1  10 19
 *          1    4                 4  13 22
 *                                 7  16 25
 *
 *                                   2  11 20
 *            2    5                 5  14 23
 *                                   8  17 26
 *
 *                                       3  12 21
 *              3    6                   6  15 24
 *                                       9  18 27
 *
 *
 * Output: 13       52
 *          28       70
 *           45       90
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {13, 28, 45, 52, 70, 90};
  auto params = get_1x1_params_fxf_filter(2, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/**
 * Input:                  Filter: 1  19 37     2  20 38
 *          1    4                 7  25 43     8  26 44
 *                                 13 31 49     14 32 50
 *
 *                                   3  21 39      4  22 40
 *            2    5                 9  27 45      10 28 46
 *                                   15 33 51      16 34 52
 *
 *                                       5  23 41     6  24 42
 *              3    6                   11 29 47     12 30 48
 *                                       17 35 53     18 36 54
 *
 *
 * Output: 25 54 87    100 135 174
 *          26 56 90    104 140 180
 *
 */
TYPED_TEST(OneByOneInputDepthConvTest, BatchedDeep1x1And3x3Filter2Features) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {25,  26,  54,  56,  87,  90,
                               100, 104, 135, 140, 174, 180};
  auto params = get_1x1_params_fxf_filter(2, 3, 2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
