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

#ifdef SNN_USE_HALF
#include <CL/sycl.hpp>
#endif  // SNN_USE_HALF

#include "test/helpers/float_comparison.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

template <typename DataType>
::testing::AssertionResult expect_not_equal(
    const char* lhs_expr, const char* rhs_expr, const char* max_ulps_expr,
    const char*, DataType const& lhs, DataType const& rhs,
    size_t const max_ulps, DataType const&) {
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
  SNN_PREDICATE_COMPARISON(expect_not_equal, expected, actual, max_ulps, 0)

template <typename DataType>
DataType get_quiet_NaN_for_type() {
  static_assert(std::numeric_limits<DataType>::is_iec559,
                "Testing code assumes IEEE 754 float representation");
  constexpr DataType qnan = std::numeric_limits<DataType>::quiet_NaN();
  return qnan;
}

template <typename DataType>
DataType get_infinity_for_type() {
  static_assert(std::numeric_limits<DataType>::is_iec559,
                "Testing code assumes IEEE 754 float representation");
  constexpr DataType inf = std::numeric_limits<DataType>::infinity();
  return inf;
}

template <typename DataType>
DataType get_negative_infinity_for_type() {
  return -(get_infinity_for_type<DataType>());
}

template <typename DataType>
DataType get_max_for_type() {
  return std::numeric_limits<DataType>::max();
}

template <typename DataType>
DataType get_lowest_for_type() {
  return std::numeric_limits<DataType>::lowest();
}

#ifdef SNN_USE_HALF
/**
 * half type as provided by SYCL cannot currently use std::numeric_limits or
 * other type traits, so we need to specialise.
 */
template <>
cl::sycl::half get_quiet_NaN_for_type() {
  return cl::sycl::half{get_quiet_NaN_for_type<float>()};
}

template <>
cl::sycl::half get_infinity_for_type() {
  auto half_inf = cl::sycl::half{get_infinity_for_type<float>()};
  return half_inf;
}

template <>
cl::sycl::half get_negative_infinity_for_type() {
  auto half_inf = cl::sycl::half{get_negative_infinity_for_type<float>()};
  return half_inf;
}

// Largest normal value for half: 0 11110 1111111111
//  = 2^(30-15) * (1 + 1 - 2^-10) = 65504
template <>
cl::sycl::half get_max_for_type() {
  return cl::sycl::half{65504};
}

// Lowest normal value for half: 1 11110 1111111111
//  = -1 * 2^(30-15) * (1 + 1 - 2^-10) = -65504
template <>
cl::sycl::half get_lowest_for_type() {
  return cl::sycl::half{-65504};
}
#endif  // SNN_USE_HALF

TEST(FloatingPointComparatorTest, Zero) {
#ifdef SNN_USE_HALF
  SNN_ALMOST_EQUAL(cl::sycl::half{0.0f}, cl::sycl::half{-0.0f}, 0);
#endif  // SNN_USE_HALF
  SNN_ALMOST_EQUAL(0.0f, -0.0f, 0);
#ifdef SNN_USE_DOUBLE
  SNN_ALMOST_EQUAL(0.0, -0.0, 0);
#endif  // SNN_USE_DOUBLE
}

/**
 * Note that signalling NaNs are treated by C++ as quiet NaNs, so no need to
 * explicitly check for them.
 */
template <typename DataType>
void test_nan_inequality() {
  auto const qnan = get_quiet_NaN_for_type<DataType>();
  ASSERT_TRUE(std::isnan(qnan));
  SNN_NOT_EQUAL(qnan, qnan, 0);
  SNN_NOT_EQUAL(qnan, qnan, 4);
}

TEST(FloatingPointComparatorTest, NaN) {
#ifdef SNN_USE_HALF
  test_nan_inequality<cl::sycl::half>();
#endif  // SNN_USE_HALF
  test_nan_inequality<float>();
#ifdef SNN_USE_DOUBLE
  test_nan_inequality<double>();
#endif  // SNN_USE_DOUBLE
}

template <typename DataType>
void test_inf_large_val_equality() {
  DataType inf = get_infinity_for_type<DataType>();
  ASSERT_TRUE(std::isinf(inf));
  DataType neg_inf = get_negative_infinity_for_type<DataType>();

  SNN_ALMOST_EQUAL(inf, inf, 0u);

  DataType near_inf = get_max_for_type<DataType>();
  SNN_ALMOST_EQUAL(inf, near_inf, 1u);

  DataType near_neg_inf = get_lowest_for_type<DataType>();
  SNN_ALMOST_EQUAL(neg_inf, near_neg_inf, 1u);
}

TEST(FloatingPointComparatorTest, LargeValueCloseToInf) {
#ifdef SNN_USE_HALF
  test_inf_large_val_equality<cl::sycl::half>();
#endif  // SNN_USE_HALF
  test_inf_large_val_equality<float>();
#ifdef SNN_USE_DOUBLE
  test_inf_large_val_equality<double>();
#endif  // SNN_USE_DOUBLE
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
    auto neg_b = DataType{-b};
    SNN_NOT_EQUAL(a, b, 4);
    SNN_ALMOST_EQUAL(a, neg_b, 4);
  }
}

TEST(FloatingPointComparatorTest, NeverEqualOppositeSign) {
#ifdef SNN_USE_HALF
  test_negatives_positives_inequality<cl::sycl::half>();
#endif  // SNN_USE_HALF
  test_negatives_positives_inequality<float>();
#ifdef SNN_USE_DOUBLE
  test_negatives_positives_inequality<double>();
#endif  // SNN_USE_DOUBLE
}

TEST(FloatingPointComparatorTest, WithinFourULPs) {
#ifdef SNN_USE_HALF
  SNN_ALMOST_EQUAL(cl::sycl::half{0.15625f}, cl::sycl::half{0.1564f}, 4);
  SNN_ALMOST_EQUAL(cl::sycl::half{0.15625f}, cl::sycl::half{0.1565f}, 4);
  SNN_ALMOST_EQUAL(cl::sycl::half{0.15625f}, cl::sycl::half{0.1566f}, 4);
  SNN_ALMOST_EQUAL(cl::sycl::half{0.15625f}, cl::sycl::half{0.1567f}, 4);
#endif  // SNN_USE_HALF

  SNN_ALMOST_EQUAL(0.15625f, 0.15625001f, 4);
  SNN_ALMOST_EQUAL(0.15625f, 0.15625003f, 4);
  SNN_ALMOST_EQUAL(0.15625f, 0.15625004f, 4);
  SNN_ALMOST_EQUAL(0.15625f, 0.15625006f, 4);

#ifdef SNN_USE_DOUBLE
  SNN_ALMOST_EQUAL(
      0.15625, 0.1562500000000000277555756156289135105907917022705078125, 4);
  SNN_ALMOST_EQUAL(0.15625,
                   0.156250000000000055511151231257827021181583404541015625, 4);
  SNN_ALMOST_EQUAL(
      0.15625, 0.1562500000000000832667268468867405317723751068115234375, 4);
  SNN_ALMOST_EQUAL(0.15625,
                   0.15625000000000011102230246251565404236316680908203125, 4);

#endif  // SNN_USE_DOUBLE
}

TEST(FloatingPointComparatorTest, NotWithinFourULPs) {
#ifdef SNN_USE_HALF
  SNN_NOT_EQUAL(cl::sycl::half{0.15625f}, cl::sycl::half{0.1569f}, 4);
#endif  // SNN_USE_HALF

  SNN_NOT_EQUAL(0.15625f, 0.15625007f, 4);

#ifdef SNN_USE_DOUBLE
  SNN_NOT_EQUAL(0.15625,
                0.1562500000000001387778780781445675529539585113525390625, 4);
#endif  // SNN_USE_DOUBLE
}

TEST(FloatingPointComparatorTest, ExactDifferenceInULPs) {
  SNN_ALMOST_EQUAL(0.15625f, -0.15625001f, 2084569090);
}

TEST(FloatingPointComparatorTest, WithinFiveULPs) {
#ifdef SNN_USE_HALF
  SNN_ALMOST_EQUAL(cl::sycl::half{0.15625f}, cl::sycl::half{0.1569f}, 5);
#endif  // SNN_USE_HALF

  SNN_ALMOST_EQUAL(0.15625f, 0.15625007f, 5);

#ifdef SNN_USE_DOUBLE
  SNN_ALMOST_EQUAL(
      0.15625, 0.1562500000000001387778780781445675529539585113525390625, 5);
#endif  // SNN_USE_DOUBLE
}

TEST(FloatingPointComparatorTest, ULPWithEps) {
#ifdef SNN_USE_HALF
  SNN_ALMOST_EQUAL_EPS(cl::sycl::half{1e-6f}, cl::sycl::half{5e-6f}, 1, 1e-5f);
#endif  // SNN_USE_HALF

  SNN_ALMOST_EQUAL_EPS(1e-6f, 5e-6f, 1, 1e-5f);

#ifdef SNN_USE_DOUBLE
  SNN_ALMOST_EQUAL_EPS(1e-6, 5e-6, 1, 1e-5f);
#endif  // SNN_USE_DOUBLE
}
