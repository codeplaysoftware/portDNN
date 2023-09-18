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

#include "portdnn/compat/pooling.hpp"
#include "test/gen/iota_initialised_data.h"

#include "portdnn/helpers/padding.h"
#include "test/helpers/float_comparison.h"

#include <type_traits>

using namespace sycldnn;
using namespace sycldnn::compat;

TEST(PoolingDesc, desc_2d_test) {
  PoolingDescriptor desc;
  constexpr int pad_h = 1, pad_w = 2;
  constexpr int stride_h = 3, stride_w = 4;
  constexpr int window_h = 5, window_w = 6;
  constexpr PoolingMode mode = PoolingMode::POOLING_MAX_DETERMINISTIC;
  constexpr NanPropagation max_pooling_nan_opt =
      NanPropagation::NOT_PROPAGATE_NAN;
  const auto status =
      setPooling2dDescriptor(desc, mode, max_pooling_nan_opt, window_h,
                             window_w, pad_h, pad_w, stride_h, stride_w);

  EXPECT_TRUE(status == StatusCode::OK);
  EXPECT_TRUE(desc.getPadH() == pad_h);
  EXPECT_TRUE(desc.getPadW() == pad_w);
  EXPECT_TRUE(desc.getStrideH() == stride_h);
  EXPECT_TRUE(desc.getStrideW() == stride_w);
  EXPECT_TRUE(desc.getWindowH() == window_h);
  EXPECT_TRUE(desc.getWindowW() == window_w);
  EXPECT_TRUE(desc.getMode() == mode);
  EXPECT_TRUE(desc.getMaxPoolNanOpt() == max_pooling_nan_opt);
}

class PoolingCompatTest : public ::testing::Test {
 protected:
  SNNHandle handle;

  void SetUp() override { SNNCreate(handle); }

  template <typename DescriptorT>
  std::pair<float*, DescriptorT> get_ptr_and_desc(
      SNNHandle& handle, const sycldnn::DataFormat format,
      const std::vector<int>& sizes) {
    DescriptorT desc;
    desc.set4d(format, sizes[0], sizes[1], sizes[2], sizes[3]);
    size_t tot_count =
        std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<int>());
    float* ptr = sycl::malloc_device<float>(tot_count, handle.getQueue());

    return std::make_pair(ptr, desc);
  }

  template <typename DType>
  void test_pool(
      const std::vector<DType>& in_data,
      const std::vector<int>& in_sizes,    // nchw
      const std::vector<int>& out_sizes,   // nchw
      const std::vector<int>& pool_sizes,  // windowhw, padhw, stridehw
      const PoolingMode pooling_mode, const NanPropagation nan_opt,
      const std::vector<float>& expect, const sycldnn::DataFormat format,
      float alpha = 1.0, float beta = 0.0) {
    PoolingDescriptor pool_desc;
    setPooling2dDescriptor(pool_desc, pooling_mode, nan_opt, pool_sizes[0],
                           pool_sizes[1], pool_sizes[2], pool_sizes[3],
                           pool_sizes[4], pool_sizes[5]);

    auto [in_ptr, in_desc] =
        get_ptr_and_desc<TensorDescriptor>(handle, format, in_sizes);
    const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                         std::multiplies<int>());
    handle.getQueue()
        .memcpy(in_ptr, in_data.data(), in_size * sizeof(float))
        .wait();

    const auto out_size = std::accumulate(out_sizes.begin(), out_sizes.end(), 1,
                                          std::multiplies<int>());

    const float max_val = 2048;
    auto output = iota_initialised_data(out_size, max_val);

    auto [out_ptr, out_desc] =
        get_ptr_and_desc<TensorDescriptor>(handle, format, out_sizes);

    handle.getQueue()
        .memcpy(out_ptr, output.data(), out_size * sizeof(float))
        .wait();

    SNNStatus status = poolingForward(handle, pool_desc, &alpha, in_desc,
                                      in_ptr, &beta, out_desc, out_ptr);

    EXPECT_EQ(status.status, StatusCode::OK);
    status.event.wait_and_throw();

    std::vector<float> out_data(out_size);
    handle.getQueue()
        .memcpy(out_data.data(), out_ptr, out_size * sizeof(float))
        .wait();

    size_t tolerance = 0u;

    // Average pooling accuracy can vary with the device used.
    if (pooling_mode == PoolingMode::POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
      tolerance = 7u;
    }

    for (size_t i = 0; i < expect.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      if (std::isnan(expect[i])) {
        EXPECT_TRUE(std::isnan(out_data[i]));
      } else {
        SNN_ALMOST_EQUAL(expect[i], out_data[i], tolerance);
      }
    }
    sycl::free(in_ptr, handle.getQueue());
    sycl::free(out_ptr, handle.getQueue());
  }
};

/*
 * Input: 1    Output: 1
 */
TEST_F(PoolingCompatTest, Basic1x1PlainAverage) {
  using DataType = float;
  const std::vector<DataType> exp_out = {1.};
  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;

  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format);
}

/*
 * Input: 1    Output: 1
 */
TEST_F(PoolingCompatTest, Basic1x1PlainMax) {
  using DataType = float;
  const std::vector<DataType> exp_out = {1.};
  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;

  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_MAX_DETERMINISTIC;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format);
}

TEST_F(PoolingCompatTest, ForwardNan1x1) {
  using DataType = float;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input = {nan};
  const std::vector<DataType> exp_out = {nan};

  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;
  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_MAX_DETERMINISTIC;
  NanPropagation max_nan_prop_opt = NanPropagation::PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;

  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format);
}

TEST_F(PoolingCompatTest, Window5Stride2SAME1x7x8x2Avg) {
  using DataType = float;
  const std::vector<DataType> exp_out = {
      20., 21., 23., 24., 27., 28., 29., 30., 36., 37., 39.,
      40., 43., 44., 45., 46., 68., 69., 71., 72., 75., 76.,
      77., 78., 84., 85., 87., 88., 91., 92., 93., 94.};
  const std::vector<int> in_sizes = {1, 2, 7, 8} /**NCHW*/;
  const auto padding_type = sycldnn::PaddingMode::SAME;
  const int window = 5;
  const int stride = 2;

  auto row_padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                         stride, padding_type);
  auto col_padding = sycldnn::helpers::calculate_padding(in_sizes[3], window,
                                                         stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], row_padding.output,
                                col_padding.output};
  std::vector<int> pool_sizes = {
      window, window, row_padding.padding, col_padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format);
}

TEST_F(PoolingCompatTest, Window7Stride4VALID1x14x14x4Max) {
  using DataType = float;
  const std::vector<DataType> exp_out = {361., 362., 363., 364., 377., 378.,
                                         379., 380., 585., 586., 587., 588.,
                                         601., 602., 603., 604.};
  const std::vector<int> in_sizes = {1, 4, 14, 14} /**NCHW*/;
  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 7;
  const int stride = 4;

  auto row_padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                         stride, padding_type);
  auto col_padding = sycldnn::helpers::calculate_padding(in_sizes[3], window,
                                                         stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], row_padding.output,
                                col_padding.output};
  std::vector<int> pool_sizes = {
      window, window, row_padding.padding, col_padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_MAX_DETERMINISTIC;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format);
}

/*
 * Input: 1    Output: 2
 */
TEST_F(PoolingCompatTest, Basic1x1PlainAverage_alpha_2_beta_0) {
  using DataType = float;
  const std::vector<DataType> exp_out = {2.};
  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;

  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  float alpha = 2.f;
  float beta = 0.f;
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format,
                            alpha, beta);
}

/*
 * Input: 1    Output: 1
 */
TEST_F(PoolingCompatTest, Basic1x1PlainAverage_alpha_0_beta_1) {
  using DataType = float;
  const std::vector<DataType> exp_out = {1.};
  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;

  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  float alpha = 0.f;
  float beta = 1.f;
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format,
                            alpha, beta);
}

/*
 * Input: 1    Output: 2
 */
TEST_F(PoolingCompatTest, Basic1x1PlainAverage_alpha_1_beta_1) {
  using DataType = float;
  const std::vector<DataType> exp_out = {2.};
  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;

  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  float alpha = 1.f;
  float beta = 1.f;
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format,
                            alpha, beta);
}

/*
 * Input: 1    Output: 2
 */
TEST_F(PoolingCompatTest, Basic1x1PlainMax_alpha_1_beta_1) {
  using DataType = float;
  const std::vector<DataType> exp_out = {2.};
  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;

  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_MAX_DETERMINISTIC;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  float alpha = 1.f;
  float beta = 1.f;
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format,
                            alpha, beta);
}

TEST_F(PoolingCompatTest, ForwardNan1x1_alpha_1_beta_1) {
  using DataType = float;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input = {nan};
  const std::vector<DataType> exp_out = {nan};

  const std::vector<int> in_sizes = {1, 1, 1, 1} /**NCHW*/;
  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 1;
  const int stride = 1;
  float alpha = 1.f;
  float beta = 1.f;

  auto padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                     stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], padding.output,
                                padding.output};
  std::vector<int> pool_sizes = {window,          window, padding.padding,
                                 padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_MAX_DETERMINISTIC;
  NanPropagation max_nan_prop_opt = NanPropagation::PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;

  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format,
                            alpha, beta);
}

TEST_F(PoolingCompatTest, Window5Stride2SAME1x7x8x2Avg_alpha_0_beta_1) {
  using DataType = float;
  const std::vector<DataType> exp_out = {
      1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11.,
      12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.,
      23., 24., 25., 26., 27., 28., 29., 30., 31., 32.};
  const std::vector<int> in_sizes = {1, 2, 7, 8} /**NCHW*/;
  const auto padding_type = sycldnn::PaddingMode::SAME;
  const int window = 5;
  const int stride = 2;

  auto row_padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                         stride, padding_type);
  auto col_padding = sycldnn::helpers::calculate_padding(in_sizes[3], window,
                                                         stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], row_padding.output,
                                col_padding.output};
  std::vector<int> pool_sizes = {
      window, window, row_padding.padding, col_padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  auto input = iota_initialised_data(in_size, max_val);
  float alpha = 0.f;
  float beta = 1.f;
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format,
                            alpha, beta);
}

TEST_F(PoolingCompatTest, Window7Stride4VALID1x14x14x4Max_alpha_0_5_beta_0_5) {
  using DataType = float;
  const std::vector<DataType> exp_out = {181., 182., 183., 184., 191., 192.,
                                         193., 194., 297., 298., 299., 300.,
                                         307., 308., 309., 310.};
  const std::vector<int> in_sizes = {1, 4, 14, 14} /**NCHW*/;
  const auto padding_type = sycldnn::PaddingMode::VALID;
  const int window = 7;
  const int stride = 4;

  auto row_padding = sycldnn::helpers::calculate_padding(in_sizes[2], window,
                                                         stride, padding_type);
  auto col_padding = sycldnn::helpers::calculate_padding(in_sizes[3], window,
                                                         stride, padding_type);
  std::vector<int> out_sizes = {in_sizes[0], in_sizes[1], row_padding.output,
                                col_padding.output};
  std::vector<int> pool_sizes = {
      window, window, row_padding.padding, col_padding.padding, stride, stride};

  PoolingMode pooling_mode = PoolingMode::POOLING_MAX_DETERMINISTIC;
  NanPropagation max_nan_prop_opt = NanPropagation::NOT_PROPAGATE_NAN;
  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC;
  const float max_val = 2048;
  const auto in_size = std::accumulate(in_sizes.begin(), in_sizes.end(), 1,
                                       std::multiplies<int>());
  float alpha = 0.5;
  float beta = 0.5;
  auto input = iota_initialised_data(in_size, max_val);
  this->test_pool<DataType>(input, in_sizes, out_sizes, pool_sizes,
                            pooling_mode, max_nan_prop_opt, exp_out, format,
                            alpha, beta);
}
