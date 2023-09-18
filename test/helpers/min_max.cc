/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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

#include "portdnn/helpers/minmax.h"

#include <stddef.h>
#include <vector>

template <typename T>
struct MinMaxTest : public ::testing::Test {
  void check_max_values(std::vector<T> const& a, std::vector<T> const& b,
                        std::vector<T> const& exp) {
    ASSERT_EQ(exp.size(), a.size());
    ASSERT_EQ(exp.size(), b.size());
    for (size_t i = 0; i < exp.size(); ++i) {
      T val = sycldnn::helpers::max(a[i], b[i]);
      EXPECT_EQ(exp[i], val);
    }
  }
  void check_min_values(std::vector<T> const& a, std::vector<T> const& b,
                        std::vector<T> const& exp) {
    ASSERT_EQ(exp.size(), a.size());
    ASSERT_EQ(exp.size(), b.size());
    for (size_t i = 0; i < exp.size(); ++i) {
      T val = sycldnn::helpers::min(a[i], b[i]);
      EXPECT_EQ(exp[i], val);
    }
  }
};
using NumericTypes = ::testing::Types<int, float>;
TYPED_TEST_SUITE(MinMaxTest, NumericTypes);

TYPED_TEST(MinMaxTest, MaxMacroNumericPositive) {
  std::vector<TypeParam> a = {5, 9, 10, 101};
  std::vector<TypeParam> b = {6, 9, 2, 100};
  std::vector<TypeParam> exp = {6, 9, 10, 101};
  this->check_max_values(a, b, exp);
}
TYPED_TEST(MinMaxTest, MaxMacroNumericNegative) {
  std::vector<TypeParam> a = {-5, -9, -10, -101};
  std::vector<TypeParam> b = {-6, -9, -2, -100};
  std::vector<TypeParam> exp = {-5, -9, -2, -100};
  this->check_max_values(a, b, exp);
}
TYPED_TEST(MinMaxTest, MinMacroNumericPositive) {
  std::vector<TypeParam> a = {5, 9, 10, 101};
  std::vector<TypeParam> b = {6, 9, 2, 100};
  std::vector<TypeParam> exp = {5, 9, 2, 100};
  this->check_min_values(a, b, exp);
}
TYPED_TEST(MinMaxTest, MinMacroNumericNegative) {
  std::vector<TypeParam> a = {-5, -9, -10, -101};
  std::vector<TypeParam> b = {-6, -9, -2, -100};
  std::vector<TypeParam> exp = {-6, -9, -10, -101};
  this->check_min_values(a, b, exp);
}
TYPED_TEST(MinMaxTest, MaxMacroPrecedence) {
  TypeParam a = sycldnn::helpers::max(1 + 2, 3);
  TypeParam a_exp = 3;
  EXPECT_EQ(a_exp, a);

  TypeParam b = sycldnn::helpers::max(1 + 2 - 2, 3);
  TypeParam b_exp = 3;
  EXPECT_EQ(b_exp, b);

  TypeParam c = sycldnn::helpers::max(4, 1 + 3);
  TypeParam c_exp = 4;
  EXPECT_EQ(c_exp, c);

  TypeParam d = sycldnn::helpers::max(4, 8 - 3);
  TypeParam d_exp = 5;
  EXPECT_EQ(d_exp, d);
}
TYPED_TEST(MinMaxTest, MinMacroPrecedence) {
  TypeParam a = sycldnn::helpers::min(1 + 2, 3);
  TypeParam a_exp = 3;
  EXPECT_EQ(a_exp, a);

  TypeParam b = sycldnn::helpers::min(1 + 2 - 2, 3);
  TypeParam b_exp = 1;
  EXPECT_EQ(b_exp, b);

  TypeParam c = sycldnn::helpers::min(4, 1 + 3);
  TypeParam c_exp = 4;
  EXPECT_EQ(c_exp, c);

  TypeParam d = sycldnn::helpers::min(4, 8 - 3);
  TypeParam d_exp = 4;
  EXPECT_EQ(d_exp, d);
}
