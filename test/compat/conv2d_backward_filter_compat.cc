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
    TensorDescriptor out_desc;
    out_desc.set4d(in_desc.getFormat(), out_n, out_c, out_h, out_w);

    auto out_data = iota_initialised_data(out_size, max_val);
    handle.getQueue()
        .memcpy(out_ptr, out_data.data(), out_size * sizeof(float))
        .wait();
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

    SNNStatus status = convolutionBackwardFilter(
        handle, &alpha, in_desc, in_ptr, out_desc, out_ptr, conv_desc,
        sycldnn::conv2d::Algorithm::Direct, nullptr, 0, &beta, filt_desc,
        filt_ptr);
    EXPECT_EQ(status.status, StatusCode::OK);
    handle.getQueue().wait();

    size_t filt_size = filt_desc.getSize();

    std::vector<float> filt_data(filt_size);
    handle.getQueue()
        .memcpy(filt_data.data(), filt_ptr, filt_size * sizeof(float))
        .wait();

    EXPECT_EQ(filt_data, expect);
    sycl::free(in_ptr, handle.getQueue());
    sycl::free(out_ptr, handle.getQueue());
    sycl::free(filt_ptr, handle.getQueue());
  }
};

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
TEST_F(Conv2DCompatTest, InputBackprop3x3) {
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, {0, 0, 1, 1, 1, 1},
                {44, 54, 64, 84, 94, 104, 124, 134, 144},
                sycldnn::DataFormat::NHWC);
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
TEST_F(Conv2DCompatTest, BatchedDeep1x1) {
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {9, 12, 15},
                sycldnn::DataFormat::NHWC);
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
TEST_F(Conv2DCompatTest, Simple1x1And3x3Filter) {
  this->do_test({1, 1, 1, 1}, {1, 1, 3, 3}, {1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 1, 0, 0, 0, 0}, sycldnn::DataFormat::NHWC);
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
TEST_F(Conv2DCompatTest, Deep1x1) {
  this->do_test({1, 3, 1, 1}, {1, 3, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 2, 3},
                sycldnn::DataFormat::NHWC);
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
TEST_F(Conv2DCompatTest, Deep1x1And3x3Filter) {
  this->do_test({1, 3, 1, 1}, {1, 3, 3, 3}, {1, 1, 1, 1, 1, 1},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,
                 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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
TEST_F(Conv2DCompatTest, FilterBackpropWindow3Stride2VALID1x5x5x1x1) {
  this->do_test({1, 1, 5, 5}, {1, 1, 3, 3}, {0, 0, 2, 2, 1, 1},
                {92., 102., 112., 142., 152., 162., 192., 202., 212.},
                sycldnn::DataFormat::NCHW);
}

TEST_F(Conv2DCompatTest, FilterBackpropWindow7Stride4SAME1x11x11x1x2) {
  this->do_test(
      {1, 1, 11, 11}, {2, 1, 7, 7}, {2, 2, 4, 4, 1, 1},
      {2820., 3016., 2872., 3072., 3956., 4250., 4028., 4328., 4100., 4406.,
       2472., 2672., 2516., 2720., 3392., 3632., 3444., 3688., 4748., 5108.,
       4820., 5186., 4892., 5264., 2956., 3200., 3000., 3248., 4008., 4302.,
       4068., 4368., 5601., 6042., 5682., 6132., 5763., 6222., 3468., 3768.,
       3516., 3822., 4668., 5028., 4728., 5094., 6492., 7032., 6573., 7122.,
       6654., 7212., 3996., 4362., 4044., 4416., 5328., 5754., 5388., 5820.,
       7383., 8022., 7464., 8112., 7545., 8202., 4524., 4956., 4572., 5010.,
       1952., 2192., 1980., 2224., 2588., 2948., 2624., 2990., 2660., 3032.,
       1492., 1736., 1512., 1760., 2260., 2544., 2288., 2576., 2984., 3410.,
       3020., 3452., 3056., 3494., 1712., 2000., 1732., 2024.},
      sycldnn::DataFormat::NHWC);
}

/**
 * Input: 1   4    Out deltas:
 *         2   5                1 2
 *          3   6
 *
 * alpha : 0.0
 * beta : 0.0
 *
 * Filter deltas:  0*(1+8)+0
 *                0*(2+10)+0
 *                0*(3+12)+0
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_0_beta_0) {
  float alpha = 0.0;
  float beta = 0.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {0, 0, 0},
                sycldnn::DataFormat::NHWC, alpha, beta);
}
/**
 * Input: 1   4    Out deltas:
 *         2   5                1 2
 *          3   6
 *
 * alpha : 0.0
 * beta : 1.0
 *
 * dw_ini: 1 2 3
 *
 * Filter deltas:  0*(1+8)+(1*1)
 *                0*(2+10)+(1*2)
 *                0*(3+12)+(1*3)
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_0_beta_1) {
  float alpha = 0.0;
  float beta = 1.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 2, 3},
                sycldnn::DataFormat::NHWC, alpha, beta);
}

/**
 * Input: 1   4    Out deltas:
 *         2   5                1 2
 *          3   6
 *
 * alpha : 1.0
 * beta : 1.0
 *
 * dw_ini: 1 2 3
 *
 * Filter deltas:  1*(1+8)+(1*1)
 *                1*(2+10)+(1*2)
 *                1*(3+12)+(1*3)
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_1_beta_1) {
  float alpha = 1.0;
  float beta = 1.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {10, 14, 18},
                sycldnn::DataFormat::NHWC, alpha, beta);
}

/**
 * Input: 1   4    Out deltas:
 *         2   5                1 2
 *          3   6
 *
 * alpha : 2.0
 * beta : 2.0
 *
 * dw_ini: 1 2 3
 *
 * Filter deltas:  2*(1+8)+(2*1)
 *                2*(2+10)+(2*2)
 *                2*(3+12)+(2*3)
 *
 */
TEST_F(Conv2DCompatTest, BatchedDeep1x1_alpha_2_beta_2) {
  float alpha = 2.0;
  float beta = 2.0;
  this->do_test({2, 1, 1, 1}, {3, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {20, 28, 36},
                sycldnn::DataFormat::NHWC, alpha, beta);
}

TEST_F(Conv2DCompatTest,
       FilterBackpropWindow7Stride4SAME1x11x11x1x2_alpha_neg_1_beta_0) {
  float alpha = -1.0;
  float beta = 0.0;
  this->do_test(
      {1, 1, 11, 11}, {2, 1, 7, 7}, {2, 2, 4, 4, 1, 1},
      {-2820., -3016., -2872., -3072., -3956., -4250., -4028., -4328., -4100.,
       -4406., -2472., -2672., -2516., -2720., -3392., -3632., -3444., -3688.,
       -4748., -5108., -4820., -5186., -4892., -5264., -2956., -3200., -3000.,
       -3248., -4008., -4302., -4068., -4368., -5601., -6042., -5682., -6132.,
       -5763., -6222., -3468., -3768., -3516., -3822., -4668., -5028., -4728.,
       -5094., -6492., -7032., -6573., -7122., -6654., -7212., -3996., -4362.,
       -4044., -4416., -5328., -5754., -5388., -5820., -7383., -8022., -7464.,
       -8112., -7545., -8202., -4524., -4956., -4572., -5010., -1952., -2192.,
       -1980., -2224., -2588., -2948., -2624., -2990., -2660., -3032., -1492.,
       -1736., -1512., -1760., -2260., -2544., -2288., -2576., -2984., -3410.,
       -3020., -3452., -3056., -3494., -1712., -2000., -1732., -2024.},
      sycldnn::DataFormat::NHWC, alpha, beta);
}
