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

#include "src/helpers/fast_div.h"

#include <stddef.h>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
struct FastDivTest : public ::testing::Test {
  void check_division_values(T const divisor, std::vector<T> const& values) {
    SCOPED_TRACE("With divisor " + std::to_string(divisor));
    sycldnn::fast_div::FastDiv<T> div(divisor);
    for (size_t i = 0; i < values.size(); ++i) {
      SCOPED_TRACE("Iteration " + std::to_string(i));
      EXPECT_EQ(values[i] / divisor, values[i] / div);
    }
  }
  void check_all_values_up_to(T const max, T const divisor) {
    SCOPED_TRACE("All values 1->" + std::to_string(max));
    std::vector<T> values{max};
    std::iota(values.begin(), values.end(), 1);
    check_division_values(divisor, values);
  }
};
using IntegerTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(FastDivTest, IntegerTypes);

TYPED_TEST(FastDivTest, DivideBy2) {
  TypeParam divisor = 2;
  std::vector<TypeParam> values = {1,  2,  3,  4,  5,   6,   7,   8,    10,
                                   11, 12, 13, 14, 100, 101, 999, 1000, 1001};
  this->check_division_values(divisor, values);
}
TYPED_TEST(FastDivTest, DivideBy3) {
  TypeParam divisor = 3;
  std::vector<TypeParam> values = {1,  2,  3,  4,  5,   6,   7,   8,    10,
                                   11, 12, 13, 14, 100, 101, 999, 1000, 1001};
  this->check_division_values(divisor, values);
}
TYPED_TEST(FastDivTest, DivideBy7) {
  TypeParam divisor = 7;
  std::vector<TypeParam> values = {1,  2,  3,  4,  5,   6,   7,   8,    10,
                                   11, 12, 13, 14, 100, 101, 999, 1000, 1001};
  this->check_division_values(divisor, values);
}
TYPED_TEST(FastDivTest, DivideBy12) {
  TypeParam divisor = 12;
  std::vector<TypeParam> values = {1,  2,  3,  4,  5,   6,   7,   8,    10,
                                   11, 12, 13, 14, 100, 101, 999, 1000, 1001};
  this->check_division_values(divisor, values);
}
TYPED_TEST(FastDivTest, AllValuesDivisorsLessThan10) {
  TypeParam max = 1024;
  for (TypeParam div = 2; div < 10; ++div) {
    this->check_all_values_up_to(max, div);
  }
}
TYPED_TEST(FastDivTest, AllValuesDivisors10To20) {
  TypeParam max = 1024;
  for (TypeParam div = 10; div < 20; ++div) {
    this->check_all_values_up_to(max, div);
  }
}
TYPED_TEST(FastDivTest, AllValuesDivisors100To200) {
  TypeParam max = 1024;
  for (TypeParam div = 100; div < 200; ++div) {
    this->check_all_values_up_to(max, div);
  }
}
