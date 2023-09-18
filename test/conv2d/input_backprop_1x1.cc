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
#include "portdnn/conv2d/params.h"

#include "test/conv2d/convolution_fixture.h"
#include "test/conv2d/selector_list.h"

#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_tuple4.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>

template <typename Pair>
using OneByOneInputConvolutionTest = ConvolutionFixture<Pair>;

using DataTypeList = sycldnn::types::KernelDataTypes;
using Selectors = sycldnn::types::SelectorList;
using Backends = sycldnn::types::DefaultBackendTypes;
using DataFormats = sycldnn::types::DataFormatTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<Selectors, DataTypeList>::type;
using BackendTypePairs =
    sycldnn::types::CartesianProduct<SNNTypePairs, Backends>::type;
using DataFormatBackendTypePairs =
    sycldnn::types::CartesianProduct<BackendTypePairs, DataFormats>::type;
using TestTuple4 =
    sycldnn::types::NestedPairsToTuple4<DataFormatBackendTypePairs>::type;

using GTestTypeTuple4s = sycldnn::types::ToGTestTypes<TestTuple4>::type;
TYPED_TEST_SUITE(OneByOneInputConvolutionTest, GTestTypeTuple4s);

sycldnn::conv2d::Conv2DParams get_1x1_params(int batch, int channels,
                                             int features) {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = channels;
  params.features = features;
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
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}

sycldnn::conv2d::Conv2DParams get_1x1_params_fxf_filter(int batch, int channels,
                                                        int features,
                                                        int window_size) {
  auto params = get_1x1_params(batch, channels, features);
  params.window_rows = window_size;
  params.window_cols = window_size;
  params.pad_rows = (window_size - 1) / 2;
  params.pad_cols = (window_size - 1) / 2;
  return params;
}

/**
 * Out deltas: 1  Input: 1
 *
 * Input deltas: 1
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Simple1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1};
  auto params = get_1x1_params(1, 1, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}
/**
 * Out deltas:  1    Input:
 *               2           1
 *                3
 *
 * Input deltas: 1
 *                2
 *                 3
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Deep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1, 2, 3};
  auto params = get_1x1_params(1, 3, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}

/**
 * Out deltas: 1   4    Filter: 1
 *              2   5            2
 *               3   6            3
 *
 * Input deltas: 1+4+9 4+10+18
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {14, 32};
  auto params = get_1x1_params(2, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}

/**
 * Out deltas: 1   4    Filter: 1 3 5
 *              2   5            2 4 6
 *               3   6
 *
 * Input deltas: 14 32
 *                32 77
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep2Channels1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {14, 32, 32, 77};
  auto params = get_1x1_params(2, 2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}
/**
 * Out deltas:     Filter:  1 2 3
 *             1            4 5 6
 *                          7 8 9
 *
 * Input deltas: 5
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Simple1x1And3x3Input) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {5};
  auto params = get_1x1_params_fxf_filter(1, 1, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}
/**
 * Out deltas:             Filter:  1  4  7
 *                                  10 13 16
 *                                  19 22 25
 *
 *                                     2  5  8
 *           1                         11 14 17
 *                                     20 23 26
 *
 *                                        3  6  9
 *                                        12 15 18
 *                                        21 24 27
 * Input deltas:  13
 *                 14
 *                  15
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Deep1x1And3x3Input) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {13, 14, 15};
  auto params = get_1x1_params_fxf_filter(1, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}
/**
 * Out deltas:             Filter:  1  4  7
 *                                  10 13 16
 *                                  19 22 25
 *
 *                                     2  5  8
 *            1     2                  11 14 17
 *                                     20 23 26
 *
 *                                        3  6  9
 *                                        12 15 18
 *                                        21 24 27
 * Input deltas:  13  26
 *                 14  28
 *                  15  30
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep1x1And3x3Input) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {13, 14, 15, 26, 28, 30};
  auto params = get_1x1_params_fxf_filter(2, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}
/**
 * Out deltas:             Filter:  1  19 37     2  20 38
 *                                  7  25 43     8  26 44
 *                                  13 31 49     14 32 50
 *
 *                                     3  21 39      4  22 40
 *            1   3                    9  27 45      10 28 46
 *             2   4                   15 33 51      16 34 52
 *
 *                                        5  23 41     6  24 42
 *                                        11 29 47     12 30 48
 *                                        17 35 53     18 36 54
 * Input deltas:  25+52 75+104
 *                 27+56  81+112
 *                  29+60  87+120
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep1x1And3x3Input2Features) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {77, 83, 89, 179, 193, 207};
  auto params = get_1x1_params_fxf_filter(2, 3, 2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}
