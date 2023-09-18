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

template <typename Tuple>
using OneByOneInputConvolutionTest = ConvolutionFixture<Tuple>;

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
 * Input: 1  Out deltas: 1
 *
 * Filter deltas: 1
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Simple1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1};
  auto params = get_1x1_params(1, 1, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}
/**
 * Input:  1    Out deltas:
 *          2                1
 *           3
 *
 * Filter deltas: 1
 *                 2
 *                  3
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Deep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1, 2, 3};
  auto params = get_1x1_params(1, 3, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}

/**
 * Input: 1   4    Out deltas:
 *         2   5                1 2
 *          3   6
 *
 * Filter deltas: (1+8)
 *                 (2+10)
 *                  (3+12)
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {9, 12, 15};
  auto params = get_1x1_params(2, 3, 1);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}

/**
 * Input: 1   4    Out deltas: 1 3
 *         2   5                2 4
 *          3   6
 *
 * Filter deltas: 1+12 2+16
 *                 2+15 4+20
 *                  3+18 6+24
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep2Features1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {13, 18, 17, 24, 21, 30};
  auto params = get_1x1_params(2, 3, 2);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}
/**
 * Input:     Out deltas:
 *         1                1
 *
 *
 * Filter deltas: 0 0 0
 *                0 1 0
 *                0 0 0
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Simple1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0, 0, 0, 0, 1, 0, 0, 0, 0};
  auto params = get_1x1_params_fxf_filter(1, 1, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}
/**
 * Input:             Out deltas: 1
 *          1
 *
 *
 *
 *            2
 *
 *
 *
 *              3
 *
 *
 *
 * Filter deltas:  0 0 0
 *                 0 1 0
 *                 0 0 0
 *
 *                   0 0 0
 *                   0 2 0
 *                   0 0 0
 *
 *                     0 0 0
 *                     0 3 0
 *                     0 0 0
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, Deep1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
                               3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto params = get_1x1_params_fxf_filter(1, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}
/**
 * Input:                  Output deltas: 1 2
 *          1    4
 *
 *
 *
 *            2    5
 *
 *
 *
 *              3    6
 *
 * Filter deltas:  0  0  0
 *                 0 1+8 0
 *                 0  0  0
 *
 *                    0  0   0
 *                    0 2+10 0
 *                    0  0   0
 *
 *                       0  0   0
 *                       0 3+12 0
 *                       0  0   0
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep1x1And3x3Filter) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 12,
                               15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto params = get_1x1_params_fxf_filter(2, 3, 1, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}
/**
 * Input:                Output deltas: 1 3
 *          1    4                       2 4
 *
 *
 *
 *            2    5
 *
 *
 *
 *              3    6
 *
 *
 *
 * Filter deltas:  0  0   0     0  0   0
 *                 0 1+12 0     0 2+16 0
 *                 0  0   0     0  0   0
 *
 *                    0  0   0     0  0   0
 *                    0 2+15 0     0 4+20 0
 *                    0  0   0     0  0   0
 *
 *                       0  0   0     0  0   0
 *                       0 3+18 0     0 6+24 0
 *                       0  0   0     0  0   0
 *
 */
TYPED_TEST(OneByOneInputConvolutionTest, BatchedDeep1x1And3x3Filter2Features) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,
                               0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 13, 18, 17, 24,
                               21, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,
                               0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0};
  auto params = get_1x1_params_fxf_filter(2, 3, 2, 3);
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}
