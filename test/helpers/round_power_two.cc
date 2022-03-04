
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

#include "src/helpers/round_power_two.h"

#include <stddef.h>
#include <cstdint>
#include <string>
#include <vector>

template <typename T>
struct PowerTwoTest : public ::testing::Test {
  void check_values(std::vector<T> const& in, std::vector<T> const& exp) {
    ASSERT_EQ(exp.size(), in.size());
    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      T val = sycldnn::helpers::round_to_power_of_two(in[i]);
      EXPECT_EQ(exp[i], val);
    }
  }
};
using NumericTypes = ::testing::Types<int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(PowerTwoTest, NumericTypes);

TYPED_TEST(PowerTwoTest, SmallValues) {
  std::vector<TypeParam> in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp = {0, 1, 2, 4, 4, 8, 8, 8, 8, 16, 16};
  this->check_values(in, exp);
}
TYPED_TEST(PowerTwoTest, Near32) {
  std::vector<TypeParam> in = {30, 31, 32, 33, 34};
  std::vector<TypeParam> exp = {32, 32, 32, 64, 64};
  this->check_values(in, exp);
}
TYPED_TEST(PowerTwoTest, Near64) {
  std::vector<TypeParam> in = {62, 63, 64, 65, 66};
  std::vector<TypeParam> exp = {64, 64, 64, 128, 128};
  this->check_values(in, exp);
}
TYPED_TEST(PowerTwoTest, Near1024) {
  std::vector<TypeParam> in = {1000, 1023, 1024, 1025, 1200};
  std::vector<TypeParam> exp = {1024, 1024, 1024, 2048, 2048};
  this->check_values(in, exp);
}
