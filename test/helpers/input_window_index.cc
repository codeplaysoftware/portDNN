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
#include "src/helpers/window_index.h"

#include <gtest/gtest.h>

#include <stddef.h>
#include <cstdint>
#include <vector>

template <typename T>
struct IndexHelpersInWindow : public ::testing::Test {
  void check_input_window(T const stride, T const pad,
                          std::vector<T> const& index,
                          std::vector<T> const& exp_win,
                          std::vector<T> const& exp_fil) {
    ASSERT_EQ(index.size(), exp_win.size());
    ASSERT_EQ(index.size(), exp_fil.size());
    for (size_t i = 0; i < index.size(); ++i) {
      auto val = sycldnn::helpers::in_window_from_output(index[i], stride, pad);
      EXPECT_EQ(exp_win[i], val.window_start);
      EXPECT_EQ(exp_fil[i], val.filter_start);
    }
  }
};
using SignedIntegralTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(IndexHelpersInWindow, SignedIntegralTypes);

TYPED_TEST(IndexHelpersInWindow, Stride1Pad0) {
  TypeParam const stride = 1;
  TypeParam const pad = 0;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride1Pad1) {
  TypeParam const stride = 1;
  TypeParam const pad = 1;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride1Pad2) {
  TypeParam const stride = 1;
  TypeParam const pad = 2;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride2Pad0) {
  TypeParam const stride = 2;
  TypeParam const pad = 0;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride2Pad1) {
  TypeParam const stride = 2;
  TypeParam const pad = 1;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride2Pad2) {
  TypeParam const stride = 2;
  TypeParam const pad = 2;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride3Pad0) {
  TypeParam const stride = 3;
  TypeParam const pad = 0;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride3Pad1) {
  TypeParam const stride = 3;
  TypeParam const pad = 1;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {-1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
TYPED_TEST(IndexHelpersInWindow, Stride3Pad2) {
  TypeParam const stride = 3;
  TypeParam const pad = 2;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {-2, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_input_window(stride, pad, index, exp_win, exp_fil);
}
