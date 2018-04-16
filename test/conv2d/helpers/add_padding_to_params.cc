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

#include "sycldnn/padding_mode.h"

#include "sycldnn/conv2d/params.h"

#include "sycldnn/conv2d/helpers/add_padding_to_params.h"

TEST(AddPaddingToParamsTest, ValidStride1) {
  sycldnn::conv2d::Conv2DParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 1;
  params.stride_cols = 1;

  auto result = sycldnn::conv2d::helpers::add_padding_to(
      params, sycldnn::PaddingMode::VALID);

  EXPECT_EQ(15, result.in_rows);
  EXPECT_EQ(10, result.in_cols);
  EXPECT_EQ(3, result.window_rows);
  EXPECT_EQ(1, result.window_cols);
  EXPECT_EQ(1, result.stride_rows);
  EXPECT_EQ(1, result.stride_cols);
  EXPECT_EQ(13, result.out_rows);
  EXPECT_EQ(10, result.out_cols);
  EXPECT_EQ(0, result.pad_rows);
  EXPECT_EQ(0, result.pad_cols);
}
TEST(AddPaddingToParamsTest, SameStride1) {
  sycldnn::conv2d::Conv2DParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 1;
  params.stride_cols = 1;

  auto result = sycldnn::conv2d::helpers::add_padding_to(
      params, sycldnn::PaddingMode::SAME);

  EXPECT_EQ(15, result.in_rows);
  EXPECT_EQ(10, result.in_cols);
  EXPECT_EQ(3, result.window_rows);
  EXPECT_EQ(1, result.window_cols);
  EXPECT_EQ(1, result.stride_rows);
  EXPECT_EQ(1, result.stride_cols);
  EXPECT_EQ(15, result.out_rows);
  EXPECT_EQ(10, result.out_cols);
  EXPECT_EQ(1, result.pad_rows);
  EXPECT_EQ(0, result.pad_cols);
}
TEST(AddPaddingToParamsTest, ValidStride2) {
  sycldnn::conv2d::Conv2DParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 2;
  params.stride_cols = 2;

  auto result = sycldnn::conv2d::helpers::add_padding_to(
      params, sycldnn::PaddingMode::VALID);

  EXPECT_EQ(15, result.in_rows);
  EXPECT_EQ(10, result.in_cols);
  EXPECT_EQ(3, result.window_rows);
  EXPECT_EQ(1, result.window_cols);
  EXPECT_EQ(2, result.stride_rows);
  EXPECT_EQ(2, result.stride_cols);
  EXPECT_EQ(7, result.out_rows);
  EXPECT_EQ(5, result.out_cols);
  EXPECT_EQ(0, result.pad_rows);
  EXPECT_EQ(0, result.pad_cols);
}
TEST(AddPaddingToParamsTest, SameStride2) {
  sycldnn::conv2d::Conv2DParams params{};
  params.in_rows = 15;
  params.in_cols = 10;
  params.window_rows = 3;
  params.window_cols = 1;
  params.stride_rows = 2;
  params.stride_cols = 2;

  auto result = sycldnn::conv2d::helpers::add_padding_to(
      params, sycldnn::PaddingMode::SAME);

  EXPECT_EQ(15, result.in_rows);
  EXPECT_EQ(10, result.in_cols);
  EXPECT_EQ(3, result.window_rows);
  EXPECT_EQ(1, result.window_cols);
  EXPECT_EQ(2, result.stride_rows);
  EXPECT_EQ(2, result.stride_cols);
  EXPECT_EQ(8, result.out_rows);
  EXPECT_EQ(5, result.out_cols);
  EXPECT_EQ(1, result.pad_rows);
  EXPECT_EQ(0, result.pad_cols);
}
