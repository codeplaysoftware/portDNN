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
struct IndexHelpersOutWindow : public ::testing::Test {
  void check_output_window(T const stride, T const pad,
                           std::vector<T> const& index,
                           std::vector<T> const& exp_win,
                           std::vector<T> const& exp_fil) {
    ASSERT_EQ(index.size(), exp_win.size());
    ASSERT_EQ(index.size(), exp_fil.size());
    for (size_t i = 0; i < index.size(); ++i) {
      auto val = sycldnn::helpers::out_window_from_input(index[i], stride, pad);
      EXPECT_EQ(exp_win[i], val.window_start);
      EXPECT_EQ(exp_fil[i], val.filter_start);
    }
  }
};
using SignedIntegralTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(IndexHelpersOutWindow, SignedIntegralTypes);

/*
 * in:  -   0   1   2   3   4
 *       \ / \ / \ / \ / \ /
 * out:   0   1   2   3   4
 */
TYPED_TEST(IndexHelpersOutWindow, Stride1Pad0) {
  TypeParam const stride = 1;
  TypeParam const pad = 0;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_fil = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/*
 * in:    0   1   2   3   4
 *       / \ / \ / \ / \ /
 * out: -   0   1   2   3
 */
TYPED_TEST(IndexHelpersOutWindow, Stride1Pad1) {
  TypeParam const stride = 1;
  TypeParam const pad = 1;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> exp_fil = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/*
 * in:    -   0   1   2   3   4
 *         \ /     \ /     \ /
 * out:     0       1       2
 */
TYPED_TEST(IndexHelpersOutWindow, Stride2Pad0) {
  TypeParam const stride = 2;
  TypeParam const pad = 0;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::vector<TypeParam> exp_fil = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/* The pad in the output is slightly odd, in that it is determined by the
 * padding without strides.
 * in:    0   1   2   3   4   5
 *       / \ /     \ /     \ /
 * out: -   0       1       2
 *
 * in:  - 0 1 2 3 4 5 6 7
 *       X|/ \|/ \|/ \|/
 * out: - 0   1   2   3
 */
TYPED_TEST(IndexHelpersOutWindow, Stride2Pad1) {
  TypeParam const stride = 2;
  TypeParam const pad = 1;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
  std::vector<TypeParam> exp_fil = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/*
 * in:  - - 0 1 2 3 4 5 6
 *       \|/ \|/ \|/ \|/
 * out:   - - 0   1   2
 */
TYPED_TEST(IndexHelpersOutWindow, Stride2Pad2) {
  TypeParam const stride = 2;
  TypeParam const pad = 2;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  std::vector<TypeParam> exp_fil = {2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/*
 * in:     - - - 0 1 2 3 4 5 6
 *        /   \|/   \|/   \|/
 * out:  - - - 0     1     2
 */
TYPED_TEST(IndexHelpersOutWindow, Stride3Pad0) {
  TypeParam const stride = 3;
  TypeParam const pad = 0;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4};
  std::vector<TypeParam> exp_fil = {0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/*
 * in:     - - 0 1 2 3 4 5 6 7
 *        /   \|/   \|/   \|/
 * out:  - - - 0     1     2
 */
TYPED_TEST(IndexHelpersOutWindow, Stride3Pad1) {
  TypeParam const stride = 3;
  TypeParam const pad = 1;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> exp_fil = {1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/*
 * in:     - 0 1 2 3 4 5 6 7 8
 *        /   \|/   \|/   \|/
 * out:  - - - 0     1     2
 */
TYPED_TEST(IndexHelpersOutWindow, Stride3Pad2) {
  TypeParam const stride = 3;
  TypeParam const pad = 2;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<TypeParam> exp_fil = {2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
/*
 * in: - - - 0 1 2 3 4 5 6 7 8
 *      \_\|/ \_\|/ \_\|/ \_\|/
 * out:    - - - 0     1     2
 */
TYPED_TEST(IndexHelpersOutWindow, Stride3Pad3) {
  TypeParam const stride = 3;
  TypeParam const pad = 3;
  std::vector<TypeParam> index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_win = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  std::vector<TypeParam> exp_fil = {3, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2};
  this->check_output_window(stride, pad, index, exp_win, exp_fil);
}
