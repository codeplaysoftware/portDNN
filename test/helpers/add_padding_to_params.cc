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

#include "portdnn/padding_mode.h"

#include "portdnn/helpers/padding.h"

#include "portdnn/conv2d/params.h"
#include "portdnn/pooling/params.h"

/** Input parameters for each test case. */
struct TestParams {
  int in_rows;
  int in_cols;
  int window_rows;
  int window_cols;
  int stride_rows;
  int stride_cols;
  int exp_out_rows;
  int exp_out_cols;
  int exp_pad_rows;
  int exp_pad_cols;
};

template <typename Params>
struct AddPaddingToParamsTest : public ::testing::Test {
  /**
   * Check that adding padding to the input test params match the expected
   * values.
   */
  void check_padding(TestParams const& test_params, sycldnn::PaddingMode mode) {
    Params snn_params{};
    snn_params.in_rows = test_params.in_rows;
    snn_params.in_cols = test_params.in_cols;
    snn_params.window_rows = test_params.window_rows;
    snn_params.window_cols = test_params.window_cols;
    snn_params.stride_rows = test_params.stride_rows;
    snn_params.stride_cols = test_params.stride_cols;

    auto result = sycldnn::helpers::add_padding_to(snn_params, mode);

    // Check the input values are not changed
    EXPECT_EQ(test_params.in_rows, result.in_rows);
    EXPECT_EQ(test_params.in_cols, result.in_cols);
    EXPECT_EQ(test_params.window_rows, result.window_rows);
    EXPECT_EQ(test_params.window_cols, result.window_cols);
    EXPECT_EQ(test_params.stride_rows, result.stride_rows);
    EXPECT_EQ(test_params.stride_cols, result.stride_cols);

    // Check the computed values are as expected
    EXPECT_EQ(test_params.exp_out_rows, result.out_rows);
    EXPECT_EQ(test_params.exp_out_cols, result.out_cols);
    EXPECT_EQ(test_params.exp_pad_rows, result.pad_rows);
    EXPECT_EQ(test_params.exp_pad_cols, result.pad_cols);
  }
};
using ParamTypes = ::testing::Types<sycldnn::conv2d::Conv2DParams,
                                    sycldnn::pooling::PoolingParams>;
TYPED_TEST_SUITE(AddPaddingToParamsTest, ParamTypes);

TYPED_TEST(AddPaddingToParamsTest, ValidStride1) {
  TestParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.exp_out_rows = 13;
  params.exp_out_cols = 10;
  params.exp_pad_rows = 0;
  params.exp_pad_cols = 0;

  this->check_padding(params, sycldnn::PaddingMode::VALID);
}
TYPED_TEST(AddPaddingToParamsTest, SameStride1) {
  TestParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.exp_out_rows = 15;
  params.exp_out_cols = 10;
  params.exp_pad_rows = 1;
  params.exp_pad_cols = 0;

  this->check_padding(params, sycldnn::PaddingMode::SAME);
}
TYPED_TEST(AddPaddingToParamsTest, ValidStride2) {
  TestParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 2;
  params.stride_cols = 2;
  params.exp_out_rows = 7;
  params.exp_out_cols = 5;
  params.exp_pad_rows = 0;
  params.exp_pad_cols = 0;

  this->check_padding(params, sycldnn::PaddingMode::VALID);
}
TYPED_TEST(AddPaddingToParamsTest, SameStride2) {
  TestParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 2;
  params.stride_cols = 2;
  params.exp_out_rows = 8;
  params.exp_out_cols = 5;
  params.exp_pad_rows = 1;
  params.exp_pad_cols = 0;

  this->check_padding(params, sycldnn::PaddingMode::SAME);
}
