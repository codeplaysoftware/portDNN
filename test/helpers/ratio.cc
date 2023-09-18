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

#include "portdnn/helpers/ratio.h"

#include <stddef.h>
#include <cstdint>
#include <vector>

template <typename T>
struct RatioHelpers : public ::testing::Test {
  void check_ratio_above_zero(T const divisor, std::vector<T> const& values,
                              std::vector<T> const& expected) {
    ASSERT_EQ(expected.size(), values.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      T val = sycldnn::helpers::round_ratio_up_above_zero(values[i], divisor);
      EXPECT_EQ(expected[i], val);
    }
  }
  void check_ratio_up(T const divisor, std::vector<T> const& values,
                      std::vector<T> const& expected) {
    ASSERT_EQ(expected.size(), values.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      T val = sycldnn::helpers::round_ratio_up(values[i], divisor);
      EXPECT_EQ(expected[i], val);
    }
  }
  void check_round_to_multiple(T const multiple, std::vector<T> const& values,
                               std::vector<T> const& expected) {
    ASSERT_EQ(expected.size(), values.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      T val =
          sycldnn::helpers::round_up_to_nearest_multiple(values[i], multiple);
      EXPECT_EQ(expected[i], val);
    }
  }
};
template <typename T>
using SignedRatioHelpers = RatioHelpers<T>;

using RatioTypes = ::testing::Types<int32_t, int64_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(RatioHelpers, RatioTypes);

using SignedRatioTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(SignedRatioHelpers, SignedRatioTypes);

TYPED_TEST(RatioHelpers, RatioUpAbove0PositiveBy1) {
  std::vector<TypeParam> num = {0, 1, 2, 3, 4, 5, 11, 101, 10001};
  TypeParam div = 1;
  std::vector<TypeParam> exp = {0, 1, 2, 3, 4, 5, 11, 101, 10001};
  this->check_ratio_above_zero(div, num, exp);
}
TYPED_TEST(RatioHelpers, RatioUpAbove0PositiveEvenBy2) {
  std::vector<TypeParam> num = {0, 2, 4, 6, 8, 26, 102, 10002};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {0, 1, 2, 3, 4, 13, 51, 5001};
  this->check_ratio_above_zero(div, num, exp);
}
TYPED_TEST(RatioHelpers, RatioUpAbove0PositiveOddBy2) {
  std::vector<TypeParam> num = {1, 3, 5, 7, 9, 27, 103, 10003};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {1, 2, 3, 4, 5, 14, 52, 5002};
  this->check_ratio_above_zero(div, num, exp);
}
TYPED_TEST(SignedRatioHelpers, RatioUpAbove0NegativeBy1) {
  std::vector<TypeParam> num = {-1, -3, -5, -7, -9, -27, -103, -10003};
  TypeParam div = 1;
  std::vector<TypeParam> exp = {0, 0, 0, 0, 0, 0, 0, 0};
  this->check_ratio_above_zero(div, num, exp);
}
TYPED_TEST(SignedRatioHelpers, RatioUpAbove0NegativeEvenBy2) {
  std::vector<TypeParam> num = {-2, -4, -6, -8, -10, -28, -104, -10004};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {0, 0, 0, 0, 0, 0, 0, 0};
  this->check_ratio_above_zero(div, num, exp);
}
TYPED_TEST(SignedRatioHelpers, RatioUpAbove0NegativeOddBy2) {
  std::vector<TypeParam> num = {-1, -3, -5, -7, -9, -27, -103, -10003};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {0, 0, 0, 0, 0, 0, 0, 0};
  this->check_ratio_above_zero(div, num, exp);
}
TYPED_TEST(RatioHelpers, RatioUpPositiveBy1) {
  std::vector<TypeParam> num = {0, 1, 2, 3, 4, 5, 11, 101, 10001};
  TypeParam div = 1;
  std::vector<TypeParam> exp = {0, 1, 2, 3, 4, 5, 11, 101, 10001};
  this->check_ratio_up(div, num, exp);
}
TYPED_TEST(RatioHelpers, RatioUpPositiveEvenBy2) {
  std::vector<TypeParam> num = {0, 2, 4, 6, 8, 26, 102, 10002};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {0, 1, 2, 3, 4, 13, 51, 5001};
  this->check_ratio_up(div, num, exp);
}
TYPED_TEST(RatioHelpers, RatioUpPositiveOddBy2) {
  std::vector<TypeParam> num = {1, 3, 5, 7, 9, 27, 103, 10003};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {1, 2, 3, 4, 5, 14, 52, 5002};
  this->check_ratio_up(div, num, exp);
}
TYPED_TEST(SignedRatioHelpers, RatioUpNegativeBy1) {
  std::vector<TypeParam> num = {-1, -3, -5, -7, -9, -27, -103, -10003};
  TypeParam div = 1;
  std::vector<TypeParam> exp = {-1, -3, -5, -7, -9, -27, -103, -10003};
  this->check_ratio_up(div, num, exp);
}
TYPED_TEST(SignedRatioHelpers, RatioUpNegativeEvenBy2) {
  std::vector<TypeParam> num = {-2, -4, -6, -8, -10, -28, -104, -10004};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {-1, -2, -3, -4, -5, -14, -52, -5002};
  this->check_ratio_up(div, num, exp);
}
TYPED_TEST(SignedRatioHelpers, RatioUpNegativeOddBy2) {
  std::vector<TypeParam> num = {-1, -3, -5, -7, -9, -27, -103, -10003};
  TypeParam div = 2;
  std::vector<TypeParam> exp = {0, -1, -2, -3, -4, -13, -51, -5001};
  this->check_ratio_up(div, num, exp);
}
TYPED_TEST(RatioHelpers, RoundMultiple1Positive) {
  std::vector<TypeParam> num = {0, 1, 2, 3, 4, 5, 11, 101, 10001};
  TypeParam multiple = 1;
  std::vector<TypeParam> exp = {0, 1, 2, 3, 4, 5, 11, 101, 10001};
  this->check_round_to_multiple(multiple, num, exp);
}
TYPED_TEST(RatioHelpers, RoundMultiple2PositiveEven) {
  std::vector<TypeParam> num = {0, 2, 4, 6, 8, 26, 102, 10002};
  TypeParam multiple = 2;
  std::vector<TypeParam> exp = {0, 2, 4, 6, 8, 26, 102, 10002};
  this->check_round_to_multiple(multiple, num, exp);
}
TYPED_TEST(RatioHelpers, RoundMultiple2PositiveOdd) {
  std::vector<TypeParam> num = {1, 3, 5, 7, 9, 27, 103, 10003};
  TypeParam multiple = 2;
  std::vector<TypeParam> exp = {2, 4, 6, 8, 10, 28, 104, 10004};
  this->check_round_to_multiple(multiple, num, exp);
}
TYPED_TEST(RatioHelpers, RoundMultiple7PositiveEven) {
  std::vector<TypeParam> num = {0, 2, 4, 6, 8, 26, 102, 10002};
  TypeParam multiple = 7;
  std::vector<TypeParam> exp = {0, 7, 7, 7, 14, 28, 105, 10003};
  this->check_round_to_multiple(multiple, num, exp);
}
TYPED_TEST(RatioHelpers, RoundMultiple7PositiveOdd) {
  std::vector<TypeParam> num = {1, 3, 5, 7, 9, 27, 103, 10003};
  TypeParam multiple = 7;
  std::vector<TypeParam> exp = {7, 7, 7, 7, 14, 28, 105, 10003};
  this->check_round_to_multiple(multiple, num, exp);
}
