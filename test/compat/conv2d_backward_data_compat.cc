#include <gtest/gtest.h>

#include <portdnn/compat/convolution.hpp>
#include <type_traits>
#include "test/gen/iota_initialised_data.h"

using namespace sycldnn;
using namespace sycldnn::compat;

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
      const FilterDescriptor& filt_desc, const ConvolutionDescriptor& conv_desc,
      const float& max_val) {
    int out_n, out_c, out_h, out_w;
    getConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n,
                                     &out_c, &out_h, &out_w);
    const size_t out_size = out_n * out_c * out_h * out_w;
    float* out_ptr = sycl::malloc_device<float>(out_size, handle.getQueue());
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
      float alpha = 1.0, float beta = 0.0) {
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

    auto [out_ptr, out_desc] =
        get_out_ptr_and_desc(handle, in_desc, filt_desc, conv_desc, max_val);

    SNNStatus status = convolutionBackwardData(
        handle, &alpha, filt_desc, filt_ptr, out_desc, out_ptr, conv_desc,
        sycldnn::conv2d::Algorithm::Direct, nullptr, 0, &beta, in_desc, in_ptr);
    EXPECT_EQ(status.status, StatusCode::OK);
    handle.getQueue().wait();

    size_t in_size = in_desc.getSize();

    std::vector<float> in_data(in_size);
    handle.getQueue()
        .memcpy(in_data.data(), in_ptr, in_size * sizeof(float))
        .wait();

    EXPECT_EQ(in_data, expect);
    sycl::free(in_ptr, handle.getQueue());
    sycl::free(out_ptr, handle.getQueue());
    sycl::free(filt_ptr, handle.getQueue());
  }
};

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
TEST_F(Conv2DCompatTest, InputBackprop3x3) {
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {1, 4, 7, 6, 7, 23, 33, 24, 19, 53, 63, 42, 21, 52, 59, 36},
                sycldnn::DataFormat::NHWC);
}

/**
 * Out deltas: 1   4    Filter: 1
 *              2   5            2
 *               3   6            3
 *
 * Input deltas: 1+4+9 4+10+18
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1) {
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {14, 32},
                sycldnn::DataFormat::NHWC);
}

/**
 * Out deltas:     Filter:  1 2 3
 *             1            4 5 6
 *                          7 8 9
 *
 * Input deltas: 5
 *
 */
TEST_F(Conv2DCompatTest, Simple1x1And3x3Input) {
  this->do_test({1, 1, 1, 1}, {1, 1, 3, 3}, {1, 1, 1, 1, 1, 1}, {5},
                sycldnn::DataFormat::NHWC);
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
TEST_F(Conv2DCompatTest, Deep1x1) {
  this->do_test({1, 3, 1, 1}, {1, 3, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 2, 3},
                sycldnn::DataFormat::NHWC);
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
TEST_F(Conv2DCompatTest, Deep1x1And3x3Input) {
  this->do_test({1, 3, 1, 1}, {1, 3, 3, 3}, {1, 1, 1, 1, 1, 1}, {13, 14, 15},
                sycldnn::DataFormat::NHWC);
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
TEST_F(Conv2DCompatTest, InputBackpropWindow3Stride2VALID1x5x5x1x1) {
  this->do_test({1, 1, 5, 5}, {1, 1, 3, 3}, {0, 0, 2, 2, 1, 1},
                {1,  2,  5,  4,  6,  4,  5,  14, 10, 12, 10, 14, 36,
                 24, 30, 12, 15, 34, 20, 24, 21, 24, 55, 32, 36},
                sycldnn::DataFormat::NCHW);
}

TEST_F(Conv2DCompatTest, InputBackpropWindow7Stride4SAME1x11x12x1x2) {
  this->do_test(
      {1, 1, 11, 11}, {2, 1, 7, 7}, {2, 2, 4, 4, 1, 1},
      {101.,  107.,  320.,  340.,  360.,  249.,  588.,  624.,  660.,  391.,
       413.,  143.,  149.,  460.,  480.,  500.,  347.,  840.,  876.,  912.,
       545.,  567.,  268.,  304.,  772.,  860.,  948.,  588.,  1308., 1428.,
       1548., 872.,  940.,  520.,  556.,  1388., 1476., 1564., 952.,  2148.,
       2268., 2388., 1348., 1416., 772.,  808.,  2004., 2092., 2180., 1316.,
       2988., 3108., 3228., 1824., 1892., 713.,  743.,  1600., 1668., 1736.,
       941.,  1980., 2064., 2148., 1139., 1185., 1072., 1156., 2380., 2564.,
       2748., 1440., 2916., 3132., 3348., 1724., 1840., 1660., 1744., 3668.,
       3852., 4036., 2140., 4428., 4644., 4860., 2536., 2652., 2248., 2332.,
       4956., 5140., 5324., 2840., 5940., 6156., 6372., 3348., 3464., 1283.,
       1337., 2740., 2856., 2972., 1535., 3120., 3252., 3384., 1733., 1803.,
       1661., 1715., 3552., 3668., 3784., 1969., 4044., 4176., 4308., 2223.,
       2293.},
      sycldnn::DataFormat::NHWC);
}

/**
 * Out deltas: 1   4    Filter: 1
 *              2   5            2
 *               3   6            3
 *
 * alpha : 0.0
 * beta : 0.0
 *
 * dx_ini: 1   2
 *
 * Input deltas: 0*(1+4+9)+0   0*(4+10+18)+0
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_0_beta_0) {
  float alpha = 0.0;
  float beta = 0.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {0, 0},
                sycldnn::DataFormat::NHWC, alpha, beta);
}
/**
 * Out deltas: 1   4    Filter: 1
 *              2   5            2
 *               3   6            3
 *
 * alpha : 0.0
 * beta : 1.0
 *
 * dx_ini: 1   2
 *
 * Input deltas: 0*(1+4+9)+(1*1)   0*(4+10+18)+(1*2)
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_0_beta_1) {
  float alpha = 0.0;
  float beta = 1.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 2},
                sycldnn::DataFormat::NHWC, alpha, beta);
}
/**
 * Out deltas: 1   4    Filter: 1
 *              2   5            2
 *               3   6            3
 *
 * alpha : 1.0
 * beta : 1.0
 *
 * dx_ini: 1   2
 *
 * Input deltas: 1*(1+4+9)+(1*1)   1*(4+10+18)+(1*2)
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_1_beta_1) {
  float alpha = 1.0;
  float beta = 1.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {15, 34},
                sycldnn::DataFormat::NHWC, alpha, beta);
}

/**
 * Out deltas: 1   4    Filter: 1
 *              2   5            2
 *               3   6            3
 *
 * alpha : 0.0
 * beta : -1.0
 *
 * dx_ini: 1   2
 *
 * Input deltas: 0*(1+4+9)+(-1*1)   0*(4+10+18)+(-1*2)
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_0_beta_neg_1) {
  float alpha = 0.0;
  float beta = -1.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {-1, -2},
                sycldnn::DataFormat::NHWC, alpha, beta);
}
/**
 * Out deltas: 1   4    Filter: 1
 *              2   5            2
 *               3   6            3
 *
 * alpha : 2.0
 * beta : 1.0
 *
 * dx_ini: 1   2
 *
 * Input deltas: 2*(1+4+9)+(1*1)   2*(4+10+18)+(1*2)
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_2_beta_1) {
  float alpha = 2.0;
  float beta = 1.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {29, 66},
                sycldnn::DataFormat::NHWC, alpha, beta);
}
/*
 * Input: 1   2  Filter:   1   2   3
 *        3   4            4   5   6
 *                         7   8   9
 *
 * alpha : 2.0
 * beta : 0.0
 *
 * dx_ini:  1   2   3   4
 *          5   6   7   8
 *          9   10  11  12
 *          13  14  15  16
 *
 * Output:   2x1x1        2x1x2         2x1x3+2x1           2x2x2
 *           2x1x4        2x1x5         2x1x6+2x4           2x2x5
 *        2x(1x7+3x1)  2x(1x8+3x2)  2x(1x9+2x7+3x3+4x1)  2x(2x8+4x2)
 *           2x3x4        2x3x5        2x(3x6+4x4)          2x4x5
 */
TEST_F(Conv2DCompatTest,
       InputBackpropWindow3Stride2VALID1x5x5x1x1_alpha_2_beta_0) {
  float alpha = 2.0;
  float beta = 0.0;
  this->do_test({1, 1, 5, 5}, {1, 1, 3, 3}, {0, 0, 2, 2, 1, 1},
                {2,  4,  10, 8,  12, 8,  10, 28, 20, 24,  20, 28, 72,
                 48, 60, 24, 30, 68, 40, 48, 42, 48, 110, 64, 72},
                sycldnn::DataFormat::NCHW, alpha, beta);
}
/*
 * Input: 1   2  Filter:   1   2   3
 *        3   4            4   5   6
 *                         7   8   9
 *
 * alpha : 0.0
 * beta : 1.0
 *
 * dx_ini:  1   2   3   4   5
 *          6   7   8   9   10
 *          11  12  13  14  15
 *          16  17  18  19  20
 *          21  22  23  24  25
 *
 * Output:  1x1      1x2        1x3+2x1        2x2
 *          1x4      1x5        1x6+2x4        2x5
 *        1x7+3x1  1x8+3x2  1x9+2x7+3x3+4x1  2x8+4x2
 *          3x4      3x5        3x6+4x4        4x5
 */
TEST_F(Conv2DCompatTest,
       InputBackpropWindow3Stride2VALID1x5x5x1x1_alpha_0_beta_1) {
  float alpha = 0.0;
  float beta = 1.0;
  this->do_test({1, 1, 5, 5}, {1, 1, 3, 3}, {0, 0, 2, 2, 1, 1},
                {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                sycldnn::DataFormat::NCHW, alpha, beta);
}
