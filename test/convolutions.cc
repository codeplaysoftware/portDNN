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
#include <gtest/gtest.h>

#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"

#include "sycldnn/conv2d/selector/direct_selector.h"

#include "test/backend/eigen_backend_test_fixture.h"

#include <CL/sycl.hpp>

#include <algorithm>
#include <vector>

struct BasicConvolutionTest : public EigenBackendTest {
 protected:
  /** Test a convolution with both input and filter set to `1, 2, 3,...` */
  template <typename ConvType>
  void test_conv(std::vector<float> exp,
                 sycldnn::conv2d::Conv2DParams const& params,
                 sycldnn::conv2d::Selector& selector) {
    auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);
    ASSERT_EQ(conv_sizes.output_size, exp.size());

    std::vector<float> input;
    iota_n(input, conv_sizes.input_size, static_cast<float>(1));

    std::vector<float> filter;
    iota_n(filter, conv_sizes.filter_size, static_cast<float>(1));

    size_t inp_bytes = conv_sizes.input_size * sizeof(exp[0]);
    float* inp_gpu = static_cast<float*>(device_.allocate(inp_bytes));
    device_.memcpyHostToDevice(inp_gpu, input.data(), inp_bytes);

    size_t fil_bytes = conv_sizes.filter_size * sizeof(exp[0]);
    float* fil_gpu = static_cast<float*>(device_.allocate(fil_bytes));
    device_.memcpyHostToDevice(fil_gpu, filter.data(), fil_bytes);

    size_t out_bytes = conv_sizes.output_size * sizeof(exp[0]);
    float* out_gpu = static_cast<float*>(device_.allocate(out_bytes));

    auto status = sycldnn::conv2d::launch<float, ConvType>(
        inp_gpu, fil_gpu, out_gpu, params, selector, backend_);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait();

    std::vector<float> output;
    output.reserve(conv_sizes.output_size);
    device_.memcpyDeviceToHost(output.data(), out_gpu, out_bytes);

    for (size_t i = 0; i < exp.size(); ++i) {
      EXPECT_FLOAT_EQ(exp[i], output[i]);
    }
  }

 private:
  /** Fill a vector with `value, value++,...` with `size` elements. */
  template <typename T>
  void iota_n(std::vector<T>& c, size_t size, T value) {
    c.reserve(size);
    std::generate_n(std::back_inserter(c), size, [&value] { return value++; });
  }
};

namespace {
cl::sycl::default_selector selector{};
}  // namespace
std::unique_ptr<Eigen::QueueInterface> EigenBackendTest::queue_interface_{
    new Eigen::QueueInterface{selector}};
Eigen::SyclDevice EigenBackendTest::device_{
    EigenBackendTest::queue_interface_.get()};
sycldnn::backend::EigenBackend EigenBackendTest::backend_{
    EigenBackendTest::device_};

sycldnn::conv2d::Conv2DParams get_3x3_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 1;
  params.features = 1;
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
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}
sycldnn::conv2d::Conv2DParams get_3x3_stride2_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 1;
  params.features = 1;
  params.batch = 1;
  params.in_rows = 4;
  params.in_cols = 4;
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 2;
  params.stride_cols = 2;
  params.out_rows = 2;
  params.out_cols = 2;
  params.pad_rows = 0;
  params.pad_cols = 0;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}
sycldnn::conv2d::Conv2DParams get_1x1_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 2;
  params.features = 2;
  params.batch = 1;
  params.in_rows = 3;
  params.in_cols = 3;
  params.window_rows = 1;
  params.window_cols = 1;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 3;
  params.out_cols = 3;
  params.pad_rows = 0;
  params.pad_cols = 0;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
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
TEST_F(BasicConvolutionTest, Simple3x3) {
  std::vector<float> exp = {348, 393, 528, 573};
  auto params = get_3x3_params();

  sycldnn::conv2d::DirectSelector direct_sel{};
  this->test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params, direct_sel);
}
/**
 *  Input:  1    3    5       Filter:  1    2
 *            2    4    6               3    4
 *
 *          7    9   11
 *            8   10   12
 *
 *         13   15   17
 *           14   16   18
 *
 *  Output:  1+6    3+12   5+18
 *             2+8    6+16  10+24
 *
 *           7+24   9+30  11+36
 *            14+32  18+40  22+48
 *
 *          14+42  15+48  17+54
 *            26+56  30+64  34+72
 */
TEST_F(BasicConvolutionTest, Simple1x1) {
  std::vector<float> exp = {7,  10, 15, 22, 23, 34, 31, 46, 39,
                            58, 47, 70, 55, 82, 63, 94, 71, 106};

  auto params = get_1x1_params();
  sycldnn::conv2d::DirectSelector direct_sel{};
  this->test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params, direct_sel);
}
/*
 * Input: 1   2  Filter:  1  2  3
 *        3   4           4  5  6
 *                        7  8  9
 *
 * Output:   1       2+2         3+4        6
 *          4+3    5+8+6+4     6+10+9+8   12+12
 *          7+12  8+14+15+16  9+16+18+20  18+24
 *          21      24+28       27+32      36
 */
TEST_F(BasicConvolutionTest, InputBackprop3x3) {
  std::vector<float> exp = {1,  4,  7,  6,  7,  23, 33, 24,
                            19, 53, 63, 42, 21, 52, 59, 36};
  auto params = get_3x3_params();
  sycldnn::conv2d::DirectSelector direct_sel{};
  this->test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp, params,
                                                             direct_sel);
}
/*
 * Input: 1   2  Filter:   1   2   3
 *        3   4            4   5   6
 *                         7   8   9
 *
 * Output:  1x1      1x2        1x3+2x1        2x2
 *          1x4      1x5        1x6+2x4        2x5
 *        1x7+3x1  1x8+3x2  1x9+2x7+3x3+4x1  2x8+4x2
 *          3x4      3x5        3x6+4x4        4x5
 */
TEST_F(BasicConvolutionTest, InputBackprop3x3Stride2) {
  std::vector<float> exp = {1,  2,  5,  4,  4,  5,  14, 10,
                            10, 14, 36, 24, 12, 15, 34, 20};
  auto params = get_3x3_stride2_params();
  sycldnn::conv2d::DirectSelector selector{};
  this->test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp, params,
                                                             selector);
}
/*
 * Input:   1    3    5   Filter:  1    2
 *           2    4    6            3    4
 *
 *          7    9   11
 *           8   10   12
 *
 *         13   15   17
 *          14   16   18
 *
 *
 * Output:  1x1+2x2     3x1+4x2     5x1+6x2
 *            1x3+2x4     3x3+4x4     5x3+6x4
 *
 *          7x1+8x2    9x1+10x2    11x1+12x2
 *            7x3+8x4   9x3+10x4     11x3+12x4
 *
 *         13x1+14x2   15x1+16x2   17x1+18x2
 *           13x3+14x4   15x3+16x4   17x3+18x4
 */
TEST_F(BasicConvolutionTest, InputBackprop1x1) {
  std::vector<float> exp = {5,  11, 11, 25, 17, 39, 23,  53, 29,
                            67, 35, 81, 41, 95, 47, 109, 53, 123};

  auto params = get_1x1_params();
  sycldnn::conv2d::DirectSelector direct_sel{};
  this->test_conv<sycldnn::conv2d::conv_type::InputBackprop>(exp, params,
                                                             direct_sel);
}
/*
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
TEST_F(BasicConvolutionTest, FilterBackprop3x3) {
  std::vector<float> exp = {44, 54, 64, 84, 94, 104, 124, 134, 144};

  auto params = get_3x3_params();
  sycldnn::conv2d::DirectSelector direct_sel{};
  this->test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp, params,
                                                              direct_sel);
}
/*
 * Input:   1    3    5   Filter:   1    3    5
 *           2    4    6             2    4    6
 *
 *          7    9   11             7    9   11
 *           8   10   12             8   10   12
 *
 *         13   15   17            13   15   17
 *          14   16   18            14   16   18
 *
 * Output: 1x1+3x3+5x5+7x7+9x9+11x11+13x13+15x15+17x17
 *           2x1+4x3+6x5+8x7+10x9+12x11+14x13+16x15+18x17
 *
 *         1x2+3x4+5x6+7x8+9x10+11x12+13x14+15x16+17x18
 *           2x2+4x4+6x6+8x8+10x10+12x12+14x14+16x16+18x18
 */
TEST_F(BasicConvolutionTest, FilterBackprop1x1) {
  std::vector<float> exp = {969, 1050, 1050, 1140};

  auto params = get_1x1_params();
  sycldnn::conv2d::DirectSelector direct_sel{};
  this->test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(exp, params,
                                                              direct_sel);
}
