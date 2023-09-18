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

#include "portdnn/compat/convolution.hpp"
#include "test/gen/iota_initialised_data.h"

#include <type_traits>

using namespace sycldnn;
using namespace sycldnn::compat;

TEST(ConvDesc, desc_2d_test) {
  ConvolutionDescriptor desc;
  constexpr int pad_h = 1, pad_w = 2;
  constexpr int stride_h = 3, stride_w = 4;
  constexpr int dilation_h = 1, dilation_w = 1;
  constexpr ConvolutionMode mode = ConvolutionMode::CROSS_CORRELATION;
  const auto status = setConvolution2dDescriptor(
      desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode);

  EXPECT_TRUE(status == StatusCode::OK);
  EXPECT_TRUE(desc.getPadH() == pad_h);
  EXPECT_TRUE(desc.getPadW() == pad_w);
  EXPECT_TRUE(desc.getStrideH() == stride_h);
  EXPECT_TRUE(desc.getStrideW() == stride_w);
  EXPECT_TRUE(desc.getDilationH() == dilation_h);
  EXPECT_TRUE(desc.getDilationW() == dilation_w);
  EXPECT_TRUE(desc.getMode() == mode);
}

TEST(ConvDesc, desc_nd_test) {
  ConvolutionDescriptor desc;
  constexpr int spatial_dims = 2;
  constexpr int pads[spatial_dims] = {1, 2};
  constexpr int strides[spatial_dims] = {3, 4};
  constexpr int dilations[spatial_dims] = {1, 1};
  constexpr ConvolutionMode mode = ConvolutionMode::CROSS_CORRELATION;
  const auto status = setConvolutionNdDescriptor(desc, spatial_dims, pads,
                                                 strides, dilations, mode);

  EXPECT_TRUE(status == StatusCode::OK);
  EXPECT_TRUE(desc.getNumDims() == spatial_dims);
  EXPECT_TRUE(desc.getPadding().size() == spatial_dims);
  EXPECT_TRUE(desc.getStride().size() == spatial_dims);
  EXPECT_TRUE(desc.getDilation().size() == spatial_dims);
  for (int i = 0; i < spatial_dims; ++i) {
    EXPECT_TRUE(desc.getPadding()[i] == pads[i]);
    EXPECT_TRUE(desc.getStride()[i] == strides[i]);
    EXPECT_TRUE(desc.getDilation()[i] == dilations[i]);
  }
  EXPECT_TRUE(desc.getMode() == mode);
}

class Conv2DCompatTest : public ::testing::Test {
 protected:
  SNNHandle handle;

  void SetUp() override { SNNCreate(handle); }

  template <typename DescriptorT>
  std::pair<float*, DescriptorT> get_ptr_and_desc(
      SNNHandle& handle, const sycldnn::DataFormat format,
      const std::vector<int>& sizes, const std::vector<float>& in_data) {
    DescriptorT desc;
    desc.set4d(format, sizes[0], sizes[1], sizes[2], sizes[3]);
    size_t tot_count =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
    float* ptr = sycl::malloc_device<float>(tot_count, handle.getQueue());
    handle.getQueue()
        .memcpy(ptr, in_data.data(), tot_count * sizeof(float))
        .wait();

    return std::make_pair(ptr, desc);
  }

  std::pair<float*, TensorDescriptor> get_out_ptr_and_desc(
      SNNHandle& handle, const TensorDescriptor& in_desc,
      const FilterDescriptor& filt_desc,
      const ConvolutionDescriptor& conv_desc) {
    int out_n, out_c, out_h, out_w;
    getConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n,
                                     &out_c, &out_h, &out_w);
    const size_t out_size = out_n * out_c * out_h * out_w;
    float* out_ptr = sycl::malloc_device<float>(out_size, handle.getQueue());
    const float max_val = 2048;
    auto out = iota_initialised_data(out_size, max_val);
    handle.getQueue()
        .memcpy(out_ptr, out.data(), out_size * sizeof(float))
        .wait();

    TensorDescriptor out_desc;
    out_desc.set4d(in_desc.getFormat(), out_n, out_c, out_h, out_w);
    return std::make_pair(out_ptr, out_desc);
  }

  void do_test(
      const std::vector<int>& in_sizes,    // nchw
      const std::vector<int>& filt_sizes,  // kchw
      const std::vector<int>& conv_sizes,  // padhw, stridehw, dilationhw
      const std::vector<float>& expect, const sycldnn::DataFormat format,
      const sycldnn::conv2d::Algorithm algo =
          sycldnn::conv2d::Algorithm::Direct,
      int groupCount = 1, float alpha = 1.0, float beta = 0.0) {
    const float max_val = 2048;
    size_t in_tot_count = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                          std::multiplies<int>());
    auto input = iota_initialised_data(in_tot_count, max_val);
    auto [in_ptr, in_desc] =
        get_ptr_and_desc<TensorDescriptor>(handle, format, in_sizes, input);

    size_t filt_tot_count = std::accumulate(
        filt_sizes.begin(), filt_sizes.end(), 1, std::multiplies<int>());
    auto filter = iota_initialised_data(filt_tot_count, max_val);
    auto [filt_ptr, filt_desc] =
        get_ptr_and_desc<FilterDescriptor>(handle, format, filt_sizes, filter);

    ASSERT_EQ(conv_sizes.size(), 6);
    ConvolutionDescriptor conv_desc;
    conv_desc.set2d(conv_sizes[0], conv_sizes[1], conv_sizes[2], conv_sizes[3],
                    conv_sizes[4], conv_sizes[5]);
    if (groupCount > 1) {
      setConvolutionGroupCount(conv_desc, groupCount);
    }
    auto [out_ptr, out_desc] =
        get_out_ptr_and_desc(handle, in_desc, filt_desc, conv_desc);

    SNNStatus status = convolutionForward(handle, &alpha, in_desc, in_ptr,
                                          filt_desc, filt_ptr, conv_desc, algo,
                                          nullptr, 0, &beta, out_desc, out_ptr);
    EXPECT_EQ(status.status, StatusCode::OK);
    handle.getQueue().wait();

    size_t out_size = out_desc.getSize();

    std::vector<float> out_data(out_size);
    handle.getQueue()
        .memcpy(out_data.data(), out_ptr, out_size * sizeof(float))
        .wait();

    EXPECT_EQ(out_data, expect);
    sycl::free(in_ptr, handle.getQueue());
    sycl::free(out_ptr, handle.getQueue());
    sycl::free(filt_ptr, handle.getQueue());
  }
};

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
TEST_F(Conv2DCompatTest, simple_3x3) {
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {348, 393, 528, 573}, sycldnn::DataFormat::NHWC);
}

/**
 * Input: 1   4    Filter: 1
 *         2   5            2
 *          3   6            3
 *
 * Output: (1+4+9) (4+10+18)
 * batch 2, channels 3, features 1
 * h 1 w 1
 */

TEST_F(Conv2DCompatTest, BatchedDeep1x1) {
  this->do_test({2, 3, 1, 1}, {1, 3, 1, 1}, {0, 0, 1, 1, 1, 1}, {14, 32},
                sycldnn::DataFormat::NHWC);
}

/**
 * Input:       Filter: 1 2 3
 *         1            4 5 6
 *                      7 8 9
 *
 * Output: 5
 *
 */
TEST_F(Conv2DCompatTest, Simple1x1And3x3Filter) {
  this->do_test({1, 1, 1, 1}, {1, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, {5},
                sycldnn::DataFormat::NHWC);
}

/**
 * Input: 1     Filter: 1
 *         2             2
 *          3             3
 *
 * Output: (1+4+9)
 *
 */
TEST_F(Conv2DCompatTest, Deep1x1) {
  this->do_test({1, 3, 1, 1}, {1, 3, 1, 1}, {0, 0, 1, 1, 1, 1}, {14},
                sycldnn::DataFormat::NHWC);
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
 * Output: (13+28+45)
 *
 */
TEST_F(Conv2DCompatTest, Deep1x1And3x3Filter) {
  this->do_test({1, 3, 1, 1}, {1, 3, 3, 3}, {1, 1, 1, 1, 1, 1}, {86},
                sycldnn::DataFormat::NHWC);
}

TEST_F(Conv2DCompatTest, ForwardWindow3Stride1) {
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {348., 393., 528., 573.}, sycldnn::DataFormat::NHWC);
}

TEST_F(Conv2DCompatTest, ForwardWindow3Stride2VALID1x5x5x1x1) {
  this->do_test({1, 1, 5, 5}, {1, 1, 3, 3}, {0, 0, 2, 2, 1, 1},
                {411., 501., 861., 951.}, sycldnn::DataFormat::NCHW);
}

TEST_F(Conv2DCompatTest, ForwardWindow7Stride4SAME1x11x11x1x2) {
  this->do_test({1, 1, 11, 11}, {2, 1, 7, 7}, {2, 2, 4, 4, 1, 1},
                {48425., 49050., 72800., 73780., 55075., 55850., 125230.,
                 127260., 177037., 180026., 126980., 129220., 91975., 94250.,
                 126210., 129500., 87825., 90250.},
                sycldnn::DataFormat::NHWC);
}

TEST_F(Conv2DCompatTest, ForwardGroup2Window2Stride1SAME1x5x5x2x2) {
  const int groupCount = 2;
  this->do_test({1, 2, 5, 5}, {2, 1, 2, 2}, {0, 0, 1, 1, 1, 1},
                {156, 204, 188, 244, 220, 284, 252, 324, 316, 404, 348,
                 444, 380, 484, 412, 524, 476, 604, 508, 644, 540, 684,
                 572, 724, 636, 804, 668, 844, 700, 884, 732, 924},
                sycldnn::DataFormat::NHWC, sycldnn::conv2d::Algorithm::Im2col,
                groupCount);
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * alpha = 2.0
 * beta = 0.0
 *
 * Output: 2*(1+4+9+20+30      2*(2+6+12+24+35
 *            +42+63+80+99)       +48+70+88+108)
 *
 *         2*(5+12+21+36+50      2*(6+14+24+40+55
 *            +66+91+112+135)       +72+98+120+144)
 */
TEST_F(Conv2DCompatTest, simple_3x3_alpha_2_beta_0) {
  const int groupCount = 1;
  float alpha = 2.0;
  float beta = 0.0;
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {696, 786, 1056, 1146}, sycldnn::DataFormat::NHWC,
                sycldnn::conv2d::Algorithm::Direct, groupCount, alpha, beta);
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * alpha = 0.0
 * beta = 0.0
 *
 * Output: 0*(1+4+9+20+30      0*(2+6+12+24+35
 *            +42+63+80+99)       +48+70+88+108)
 *
 *         0*(5+12+21+36+50    0*(6+14+24+40+55
 *            +66+91+112+135)     +72+98+120+144)
 */
TEST_F(Conv2DCompatTest, simple_3x3_alpha_0_beta_0) {
  const int groupCount = 1;
  float alpha = 0.0;
  float beta = 0.0;
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1}, {0, 0, 0, 0},
                sycldnn::DataFormat::NHWC, sycldnn::conv2d::Algorithm::Direct,
                groupCount, alpha, beta);
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * y_ini:  1 2 3 4
 *
 * alpha = -2.0
 * beta = 0.0
 *
 * Output: -2*(1+4+9+20+30             -2*(2+6+12+24+35
 *            +42+63+80+99)+(0*1)       +48+70+88+108)+(0*2)
 *
 *         -2*(5+12+21+36+50           -2*(6+14+24+40+55
 *            +66+91+112+135)+(0*3)     +72+98+120+144)+(0*4)
 */
TEST_F(Conv2DCompatTest, simple_3x3_alpha_neg_2_beta_0) {
  const int groupCount = 1;
  float alpha = -2.0;
  float beta = 0.0;
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {-696, -786, -1056, -1146}, sycldnn::DataFormat::NHWC,
                sycldnn::conv2d::Algorithm::Direct, groupCount, alpha, beta);
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * y_ini:  1 2 3 4
 *
 * alpha = 1.0
 * beta = 1.0
 *
 * Output: 1*(1+4+9+20+30             1*(2+6+12+24+35
 *            +42+63+80+99)+(1*1)       +48+70+88+108)+(1*2)
 *
 *         1*(5+12+21+36+50           1*(6+14+24+40+55
 *            +66+91+112+135)+(1*3)     +72+98+120+144)+(1*4)
 */
TEST_F(Conv2DCompatTest, simple_3x3_alpha_1_beta_1) {
  const int groupCount = 1;
  float alpha = 1.0;
  float beta = 1.0;
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {349, 395, 531, 577}, sycldnn::DataFormat::NHWC,
                sycldnn::conv2d::Algorithm::Direct, groupCount, alpha, beta);
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * y_ini:  1 2 3 4
 *
 * alpha = 0.0
 * beta = 1.0
 *
 * Output: 0*(1+4+9+20+30             0*(2+6+12+24+35
 *            +42+63+80+99)+(1*1)       +48+70+88+108)+(1*2)
 *
 *         0*(5+12+21+36+50           0*(6+14+24+40+55
 *            +66+91+112+135)+(1*3)     +72+98+120+144)+(1*4)
 */
TEST_F(Conv2DCompatTest, simple_3x3_alpha_0_beta_1) {
  const int groupCount = 1;
  float alpha = 0.0;
  float beta = 1.0;
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1}, {1, 2, 3, 4},
                sycldnn::DataFormat::NHWC, sycldnn::conv2d::Algorithm::Direct,
                groupCount, alpha, beta);
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * y_ini:  1 2 3 4
 *
 * alpha = 0.0
 * beta = 2.0
 *
 * Output: 0*(1+4+9+20+30             0*(2+6+12+24+35
 *            +42+63+80+99)+(2*1)       +48+70+88+108)+(2*2)
 *
 *         0*(5+12+21+36+50           0*(6+14+24+40+55
 *            +66+91+112+135)+(2*3)     +72+98+120+144)+(2*4)
 */
TEST_F(Conv2DCompatTest, simple_3x3_alpha_0_beta_2) {
  const int groupCount = 1;
  float alpha = 0.0;
  float beta = 2.0;
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1}, {2, 4, 6, 8},
                sycldnn::DataFormat::NHWC, sycldnn::conv2d::Algorithm::Direct,
                groupCount, alpha, beta);
}

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 *
 * y_ini:  1 2 3 4
 *
 * alpha = 0.0
 * beta = -2.0
 *
 * Output: 0*(1+4+9+20+30             0*(2+6+12+24+35
 *            +42+63+80+99)+(-2*1)       +48+70+88+108)+(-2*2)
 *
 *         0*(5+12+21+36+50           0*(6+14+24+40+55
 *            +66+91+112+135)+(-2*3)     +72+98+120+144)+(-2*4)
 */
TEST_F(Conv2DCompatTest, simple_3x3_alpha_0_beta_neg_2) {
  const int groupCount = 1;
  float alpha = 0.0;
  float beta = -2.0;
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {-2, -4, -6, -8}, sycldnn::DataFormat::NHWC,
                sycldnn::conv2d::Algorithm::Direct, groupCount, alpha, beta);
}

TEST_F(Conv2DCompatTest, SetGetGroupCount) {
  ConvolutionDescriptor conv_desc;
  setConvolutionGroupCount(conv_desc, 5);
  EXPECT_EQ(conv_desc.getGroupCount(), 5);
}
