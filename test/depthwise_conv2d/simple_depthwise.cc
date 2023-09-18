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

#include "portdnn/backend/snn_backend.h"

#include "portdnn/depthwise_conv2d/params.h"

#include "test/depthwise_conv2d/depthwise_conv2d_fixture.h"

#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/to_gtest_types.h"
#include "test/types/type_list.h"

#include <vector>

template <typename Pair>
using BasicConvolutionTest =
    sycldnn::depthwise_conv2d::DepthwiseConv2DFixture<Pair>;

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::TypeList<sycldnn::backend::SNNBackend>;

using BackendTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;
using GTestTypePairs = sycldnn::types::ToGTestTypes<BackendTypePairs>::type;
TYPED_TEST_SUITE(BasicConvolutionTest, GTestTypePairs);

sycldnn::depthwise_conv2d::DepthwiseConv2DParams get_3x3_params() {
  sycldnn::depthwise_conv2d::DepthwiseConv2DParams params;
  params.channels = 1;
  params.channel_multiplier = 1;
  params.batch = 1;
  params.in_rows = 4;
  params.in_cols = 4;
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 2;
  params.out_cols = 2;
  params.pad_rows = 0;
  params.pad_cols = 0;
  return params;
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * Output: (1+4+9+20+30      (2+6+12+24+35
 *         +42+63+80+99)     +48+70+88+108)
 *
 *         (5+12+21+36+50    (6+14+24+40+55
 *         +66+91+112+135)   +72+98+120+144)
 */
TYPED_TEST(BasicConvolutionTest, Simple3x3) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {348, 393, 528, 573};
  auto params = get_3x3_params();
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params);
}
/*
 * For the input backprop the input is the tensor of errors to propagate. The
 * output is the tensor of propagated errors. The filter is the original filter
 * tensor.
 *
 * The input is the same size as the original convolution's output, and the
 * output is the size of the original convolution's input.
 *
 * Input: 1   2  Filter:  1  2  3
 *        3   4           4  5  6
 *                        7  8  9
 *
 * Output:   1       2+2         3+4        6
 *          4+3    5+8+6+4     6+10+9+8   12+12
 *          7+12  8+14+15+16  9+16+18+20  18+24
 *          21      24+28       27+32      36
 */
TYPED_TEST(BasicConvolutionTest, InputBackprop3x3) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1,  4,  7,  6,  7,  23, 33, 24,
                               19, 53, 63, 42, 21, 52, 59, 36};
  auto params = get_3x3_params();
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp,
                                                                      params);
}
/*
 * For the filter backprop, the input is the original input tensor and the
 * filter is the tensor of errors to propagate (with size matching the original
 * output tensor). The output is the errors in the filter, and so has the same
 * size as the original filter tensor.
 *
 * In this test case we have two channels in the input, shown with an indent,
 * and so the original output (and hence the filter backprop's filter) also has
 * two channels.
 *
 * Input:   1    3    5   Filter:  1    2
 *           2    4    6            3    4
 *
 *          7    9   11
 *           8   10   12
 *
 *         13   15   17
 *          14   16   18
 *
 * Output:   1x1+2x2+5x3+6x4      2x1+3x2+6x3+7x4       3x1+4x2+7x3+8x4
 *           5x1+6x29x3+10x4     6x1+7x2+10x3+11x4     7x1+8x2+11x3+12x4
 *         9x1+10x2+13x3+14x4   10x1+11x2+14x3+15x4   11x1+12x2+15x3+16x4
 */
TYPED_TEST(BasicConvolutionTest, FilterBackprop3x3) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {44, 54, 64, 84, 94, 104, 124, 134, 144};

  auto params = get_3x3_params();
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp,
                                                                       params);
}
