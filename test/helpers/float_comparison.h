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

/**
 * This code mimics GoogleTest in using ULPs for comparisons between floating-
 * point types. ULPs (Units in the Last Place) equals the no. of representable
 * floats in between same-sign inputs. For example, using a 32-bit IEEE float:
 *
 *  sign   | exponent | fraction
 *    0    | 01111100 | 01000000000000000000000 = 0.15625
 *    0    | 01111100 | 01000000000000000000001 = 0.15625001
 *
 * There are 0 representable floats between 0.15625 and itself, so its
 * difference in ULPs with itself == 0. Assuming a threshold of 4 ULPs,
 * 0.15625 == 0.15625001 is true, since it is the next representable float
 * from 0.15625 (difference in ULPs == 1). However, 0.15625 == 0.15625007 is
 * false, since there are 5 representable floats between them.
 *
 * This example generalises to double-precision as well, accounting for the
 * different sizes of the exponent and fraction. This accounting for sizes
 * of types is implemented by the FloatingPointConfig struct.
 *
 * This comparison is implemented by interpreting floats as their unique
 * signed-magnitude integer equivalents, and taking their absolute difference.
 * This difference is equivalent to their difference in ULPs.
 *
 * For further details, see this article for details on ULP:
 * http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 */

#ifndef PORTDNN_TEST_HELPERS_FLOAT_COMPARISON_H_
#define PORTDNN_TEST_HELPERS_FLOAT_COMPARISON_H_

#ifdef SNN_USE_HALF
#include <CL/sycl.hpp>
#endif  // SNN_USE_HALF

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

template <int num_bytes>
struct TypeWithSize;

/** Integer type for representing 16-bit floating point. */
template <>
struct TypeWithSize<2> {
  using Type = uint16_t;
};

/** Integer type for representing 32-bit floating point. */
template <>
struct TypeWithSize<4> {
  using Type = uint32_t;
};

/** Integer type for representing 64-bit floating point. */
template <>
struct TypeWithSize<8> {
  using Type = uint64_t;
};

/** The number of explicit bits used to represent the fraction. */
template <typename DataType>
struct NumFractionBits {
  static constexpr size_t value = std::numeric_limits<DataType>::digits - 1;
};

#ifdef SNN_USE_HALF
template <>
struct NumFractionBits<cl::sycl::half> {
  static constexpr size_t value = 10;
};
#endif  // SNN_USE_HALF

/** Provide a safe type that can be piped to an ostream. */
template <typename DataType>
struct Printable {
  using type = DataType;
};

#ifdef SNN_USE_HALF
// Trying to pipe a half to an ostream can cause ambiguous function errors, so
// explicitly use float as the printable type.
template <>
struct Printable<cl::sycl::half> {
  using type = float;
};
#endif  // SNN_USE_HALF

/**
 * Struct for handling exponent/fraction sizes for different types like
 * float/double. General pattern of an IEEE floating-point type is:
 *
 *  sign-bit | exponent-bits | fraction-bits
 *
 * For all types, the sign requires one bit.
 * For 32-bit floats, the exponent and fraction use 8 and 23 bits respectively.
 * For 64-bit doubles, the exponent and fraction use 11 and 52 bits
 * respectively.
 */
template <typename DataType>
struct FloatingPointConfig {
  using RawBits = typename TypeWithSize<sizeof(DataType)>::Type;

  static constexpr size_t num_bits_ = CHAR_BIT * sizeof(DataType);
  static constexpr size_t num_fraction_bits_ = NumFractionBits<DataType>::value;
  static constexpr size_t num_exponent_bits_ =
      num_bits_ - num_fraction_bits_ - 1;

  /** A variable with 1 for the leftmost bit, and all other bits 0. */
  static constexpr RawBits sign_mask_ = static_cast<RawBits>(1u)
                                        << (num_bits_ - 1);
  /** A variable with all bits set to 1s. */
  static constexpr RawBits all_bits_set = ~static_cast<RawBits>(0);
  /** A variable with all 1s shifted off to the right by the number of bits in
   * the exponent and sign. */
  static constexpr RawBits fraction_mask_ = all_bits_set >>
                                            (num_exponent_bits_ + 1);
  /** A variable with all bits that are neither in the sign, nor fraction, set
   * to 1s. */
  static constexpr RawBits exponent_mask_ =
      static_cast<RawBits>(~(sign_mask_ | fraction_mask_));
};

/**
 * Wrapper for interpreting floating-point values as their signed-magnitude
 * integer representations. Signed-magnitude integers are described as follows:
 *
 *  sign | magnitude (powers of 2 counting from right to left)
 *   0   | 0111110001000000000000000000000 = 1042284544
 *   1   | 0111110001000000000000000000001 = -1105199103
 *
 * ULPs is equivalent to the absolute difference between those integers.
 */
template <typename DataType>
struct FloatingPoint {
  using Config = FloatingPointConfig<DataType>;
  using RawBits = typename Config::RawBits;

  /**
   * memcpy-ing val rather than casting preserves the same underlying
   * bit-pattern.
   */
  explicit FloatingPoint(DataType val) {
    std::memcpy(&fp_variable_, &val, sizeof(DataType));
  }

  RawBits bits() const { return fp_variable_; }

  RawBits exponent_bits() const {
    return Config::exponent_mask_ & fp_variable_;
  }

  RawBits fraction_bits() const {
    return Config::fraction_mask_ & fp_variable_;
  }

  /**
   * IEEE standard defines NaN as all values with all 1s in exponent and
   * non-zero fraction.
   */
  bool is_NaN() const {
    return (exponent_bits() == Config::exponent_mask_) &&
           (fraction_bits() != 0);
  }

 private:
  /**
   * Although stored using an unsigned integer, fp_variable should initially
   * be interpreted as its unique signed-magnitude integer.
   */
  RawBits fp_variable_;
};

/**
 * This function takes inputs interpreted as signed-magnitude integers
 * and offsets them by 2^(num_bits_ - 1), giving their biased representation
 * which removes signedness. The biased representation is given by:
 *
 *  > For positive numbers, add 2^31
 *  > For negative numbers, find its positive equivalent in unsigned 2's
 *    complement.
 *
 * As an example:
 *
 *   Bit representation of 0.15625f:
 *   10111110001000000000000000000001 = -1105199103 (in signed-magnitude form)
 *-> 01000001110111111111111111111111 = 1105199103 (now in biased form)
 *
 *   Bit representation of -0.15625001f:
 *   00111110001000000000000000000001 = 1042284545 (in signed-magnitude form)
 *-> 10111110001000000000000000000001 = 3189768193 (now in biased form)
 *
 * diff. in ULPs: max(1105199103, 3189768193) - min(1105199103, 3189768193)
 *               = 3189768193 - 1105199103
 *               = 2084569090
 *
 * So 0.15625f and -0.15625001f have 2084569090 representable floats between
 * them.
 *
 * Note that overflow is handled because we directly manipulate the existing
 * bits. This biased form is effectively unsigned, which will allow us to
 * do subtraction safely.
 */
template <typename DataType>
inline typename FloatingPoint<DataType>::RawBits signed_to_biased(
    FloatingPoint<DataType> const& val) {
  using Config = FloatingPointConfig<DataType>;
  auto const signed_magnitude_val = val.bits();
  if (Config::sign_mask_ & signed_magnitude_val) {
    // Input is negative, add (2^num_bits - 1) by negating and adding 1.
    return (~signed_magnitude_val + 1);
  } else {
    // Input is positive, add (2^num_bits - 1) by setting most-significant bit.
    return (Config::sign_mask_ | signed_magnitude_val);
  }
}

/**
 * Convert signed-magnitude inputs to biased form and return the
 * difference between the results. This ensures no overflow when taking
 * the difference.
 */
template <typename DataType>
inline typename FloatingPoint<DataType>::RawBits unsigned_difference(
    FloatingPoint<DataType> const& x, FloatingPoint<DataType> const& y) {
  auto const biased_x = signed_to_biased(x);
  auto const biased_y = signed_to_biased(y);
  return (biased_x >= biased_y) ? (biased_x - biased_y) : (biased_y - biased_x);
}

/**
 * ULP and epsilon comparison between two floating point values, returning a
 * formatted message on returning false. The 2 comparison methods must fail to
 * fail the test. Note that:
 *
 *  - For all x, x != -x, except 0 and -0 (as per IEEE standard)
 *  - Any comparison involving NaNs returns false (as per IEEE standard)
 *  - A sufficiently large number will be nearly equal to infinity.
 *  - A sufficiently small number (denorms) will be nearly equal to 0.
 *  - For small numbers below epsilon, ULP comparison can be too strict hence
 *    it is mitigated with epsilon comparison.
 */
template <typename DataType>
inline ::testing::AssertionResult expect_almost_equal(
    const char* lhs_expr, const char* rhs_expr, const char* max_ulps_expr,
    const char* eps_expr, DataType const& lhs, DataType const& rhs,
    size_t const max_ulps, DataType const& eps) {
  using PrintableType = typename Printable<DataType>::type;
  FloatingPoint<DataType> x(lhs);
  FloatingPoint<DataType> y(rhs);

  auto difference_in_ulps = unsigned_difference(x, y);
  auto diff_eps = fabs(double(lhs) - double(rhs));

  if (x.is_NaN() || y.is_NaN() ||
      (difference_in_ulps > max_ulps && diff_eps > eps)) {
    PrintableType print_lhs = lhs;
    PrintableType print_rhs = rhs;
    PrintableType print_eps = eps;
    return ::testing::AssertionFailure()
           << "  expected: " << lhs_expr << " (" << print_lhs << "), "
           << "actual: " << rhs_expr << " (" << print_rhs << "), "
           << "ULPs: " << difference_in_ulps << " when testing with "
           << max_ulps_expr << " (" << max_ulps << "), "
           << "epsilon: " << diff_eps << " when testing with " << eps_expr
           << " (" << print_eps << ")";
  } else {
    return ::testing::AssertionSuccess();
  }
}

#define GET_UNQUALIFIED_T(input) \
  typename std::remove_cv<       \
      typename std::remove_reference<decltype(input)>::type>::type

#define SNN_PREDICATE_COMPARISON(formatted_predicate, expected, actual, \
                                 max_ulps, epsilon)                     \
  static_assert(std::is_same<GET_UNQUALIFIED_T(expected),               \
                             GET_UNQUALIFIED_T(actual)>::value,         \
                "expected and actual must be of the same type!\n");     \
  EXPECT_PRED_FORMAT4(formatted_predicate<GET_UNQUALIFIED_T(expected)>, \
                      expected, actual, max_ulps, epsilon)

/**
 * Usage: replace GoogleTest's EXPECT_{FLOAT|DOUBLE}_EQ(exp[i], out[i]) with
 * SNN_ALMOST_EQUAL(exp[i], out[i], num_ulps). GoogleTest by default uses
 * num_ulps = 4. No type specialisation is required.
 */
#define SNN_ALMOST_EQUAL(expected, actual, max_ulps) \
  SNN_PREDICATE_COMPARISON(expect_almost_equal, expected, actual, max_ulps, 0)

/**
 * Usage: replace GoogleTest's EXPECT_{FLOAT|DOUBLE}_EQ(exp[i], out[i]) with
 * SNN_ALMOST_EQUAL(exp[i], out[i], num_ulps, eps). GoogleTest by default uses
 * num_ulps = 4. No type specialisation is required.
 */
#define SNN_ALMOST_EQUAL_EPS(expected, actual, max_ulps, eps) \
  SNN_PREDICATE_COMPARISON(expect_almost_equal, expected, actual, max_ulps, eps)

#endif  // PORTDNN_TEST_HELPERS_FLOAT_COMPARISON_H_
