#include <gtest/gtest.h>

#include <sycldnn/compat/convolution.hpp>
#include <type_traits>
#include "test/gen/iota_initialised_data.h"

using namespace sycldnn;
using namespace sycldnn::compat;

class Conv2DCompatTest : public ::testing::Test {
 protected:
  SNNHandle handle;

  void SetUp() override { SNNCreate(handle); }

  size_t mul_all(const std::vector<int>& v) {
    size_t res = 1;
    for (auto& el : v) {
      SNN_ASSERT(el > 0, "Non strictly positive index value");
      res *= el;
    }
    return res;
  }

  template <typename DescriptorT>
  std::pair<float*, DescriptorT> get_ptr_and_desc(
      SNNHandle& handle, const sycldnn::DataFormat format,
      const std::vector<int>& sizes, const std::vector<float>& in_data) {
    DescriptorT desc;
    desc.set4d(format, sizes[0], sizes[1], sizes[2], sizes[3]);
    size_t tot_count = mul_all(sizes);
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
    TensorDescriptor out_desc;
    out_desc.set4d(in_desc.getFormat(), out_n, out_c, out_h, out_w);
    return std::make_pair(out_ptr, out_desc);
  }

  void do_test(
      const std::vector<int>& in_sizes,    // nchw
      const std::vector<int>& filt_sizes,  // kchw
      const std::vector<int>& conv_sizes,  // padhw, stridehw, dilationhw
      const std::vector<float>& expect, const sycldnn::DataFormat format) {
    const float max_val = 2048;
    auto input = iota_initialised_data(mul_all(in_sizes), max_val);
    auto [in_ptr, in_desc] =
        get_ptr_and_desc<TensorDescriptor>(handle, format, in_sizes, input);

    auto filter = iota_initialised_data(mul_all(filt_sizes), max_val);
    auto [filt_ptr, filt_desc] =
        get_ptr_and_desc<FilterDescriptor>(handle, format, filt_sizes, filter);

    ASSERT_EQ(conv_sizes.size(), 6);
    ConvolutionDescriptor conv_desc;
    conv_desc.set2d(conv_sizes[0], conv_sizes[1], conv_sizes[2], conv_sizes[3],
                    conv_sizes[4], conv_sizes[5]);

    auto [out_ptr, out_desc] =
        get_out_ptr_and_desc(handle, in_desc, filt_desc, conv_desc);

    SNNStatus status = convolutionForward(
        handle, /*alpha*/ nullptr, in_desc, in_ptr, filt_desc, filt_ptr,
        conv_desc, sycldnn::conv2d::Algorithm::Direct, nullptr, 0,
        /*beta*/ nullptr, out_desc, out_ptr);
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
