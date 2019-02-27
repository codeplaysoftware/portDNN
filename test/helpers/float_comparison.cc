/*
 * Copyright 2019 Codeplay Software Ltd.
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

#include "test/helpers/float_comparison.h"

#include "gtest/gtest.h"

#include <limits>

template <typename DataType>
::testing::AssertionResult expect_not_equal(
    const char* lhs_expr, const char* rhs_expr, const char* max_ulps_expr,
    DataType const& lhs, DataType const& rhs, size_t const max_ulps) {
  FloatingPoint<DataType> x(lhs);
  FloatingPoint<DataType> y(rhs);

  auto difference_in_ulps = unsigned_difference(x, y);

  if (!(x.is_NaN()) && !(y.is_NaN()) && difference_in_ulps <= max_ulps) {
    return ::testing::AssertionFailure()
           << "expected: " << lhs_expr << " (" << lhs << "), "
           << "actual: " << rhs_expr << " (" << rhs << "), "
           << "ULPs: " << difference_in_ulps << " when testing with "
           << max_ulps_expr << " (" << max_ulps << ")";
  } else {
    return ::testing::AssertionSuccess();
  }
}

#define SNN_NOT_EQUAL(expected, actual, max_ulps) \
  SNN_PREDICATE_COMPARISON(expect_not_equal, expected, actual, max_ulps)

TEST(FloatingPointComparatorTest, Zero) {
  SNN_ALMOST_EQUAL(0.0f, -0.0f, 0);
  SNN_ALMOST_EQUAL(0.0, -0.0, 0);
}

/**
 * Note that signalling NaNs are treated by C++ as quiet NaNs, so no need to
 * explicitly check for them.
 */
template <typename DataType>
void test_nan_inequality() {
  static_assert(std::numeric_limits<DataType>::is_iec559,
                "Testing code assumes IEEE 754 float representation");
  constexpr DataType qnan = std::numeric_limits<DataType>::quiet_NaN();
  static_assert(std::isnan(qnan), "Plaform does not provide a quiet NaN");
  SNN_NOT_EQUAL(qnan, qnan, 0);
  SNN_NOT_EQUAL(qnan, qnan, 4);
}

TEST(FloatingPointComparatorTest, NaN) {
  test_nan_inequality<float>();
  test_nan_inequality<double>();
}

template <typename DataType>
void test_inf_large_val_equality() {
  static_assert(std::numeric_limits<DataType>::is_iec559,
                "Testing code assumes IEEE 754 float representation");
  constexpr DataType my_inf = std::numeric_limits<DataType>::infinity();
  SNN_ALMOST_EQUAL(my_inf, my_inf, 0u);

  DataType near_inf = std::numeric_limits<DataType>::max();
  SNN_ALMOST_EQUAL(my_inf, near_inf, 1u);

  DataType near_neg_inf = std::numeric_limits<DataType>::lowest();
  SNN_ALMOST_EQUAL(-my_inf, near_neg_inf, 1u);
}

TEST(FloatingPointComparatorTest, LargeValueCloseToInf) {
  test_inf_large_val_equality<float>();
  test_inf_large_val_equality<double>();
}

template <typename DataType>
void test_negatives_positives_inequality() {
  std::vector<DataType> positives(100);
  std::iota(positives.begin(), positives.end(), 1);

  std::vector<DataType> negatives(100);
  std::iota(negatives.begin(), negatives.end(), -100);
  std::reverse(negatives.begin(), negatives.end());

  for (size_t i = 0; i < positives.size(); ++i) {
    auto a = positives[i];
    auto b = negatives[i];
    SNN_NOT_EQUAL(a, b, 4);
    SNN_ALMOST_EQUAL(a, -b, 4);
  }
}

TEST(FloatingPointComparatorTest, NeverEqualOppositeSign) {
  test_negatives_positives_inequality<float>();
  test_negatives_positives_inequality<double>();
}

TEST(FloatingPointComparatorTest, WithinFourULPs) {
  SNN_ALMOST_EQUAL(0.15625f, 0.15625001f, 4);
  SNN_ALMOST_EQUAL(0.15625f, 0.15625003f, 4);
  SNN_ALMOST_EQUAL(0.15625f, 0.15625006f, 4);

  SNN_ALMOST_EQUAL(
      0.15625, 0.1562500000000000277555756156289135105907917022705078125, 4);
  SNN_ALMOST_EQUAL(0.15625,
                   0.156250000000000055511151231257827021181583404541015625, 4);
  SNN_ALMOST_EQUAL(0.15625,
                   0.15625000000000011102230246251565404236316680908203125, 4);
}

TEST(FloatingPointComparatorTest, NotWithinFourULPs) {
  SNN_NOT_EQUAL(0.15625f, 0.15625007f, 4);

  SNN_NOT_EQUAL(0.15625,
                0.1562500000000001387778780781445675529539585113525390625, 4);
}

TEST(FloatingPointComparatorTest, ExactDifferenceInULPs) {
  SNN_ALMOST_EQUAL(0.15625f, -0.15625001f, 2084569090);
}

TEST(FloatingPointComparatorTest, WithinFiveULPs) {
  SNN_ALMOST_EQUAL(0.15625f, 0.15625007f, 5);

  SNN_ALMOST_EQUAL(
      0.15625, 0.1562500000000001387778780781445675529539585113525390625, 5);
}
