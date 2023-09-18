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
#ifndef PORTDNN_INCLUDE_HELPERS_RATIO_H_
#define PORTDNN_INCLUDE_HELPERS_RATIO_H_

/**
 * \file
 * Contains helper functions to control rounding behaviour in integer division.
 */
#include "portdnn/helpers/macros.h"
#include "portdnn/helpers/non_deduced_type.h"

#include <type_traits>

namespace sycldnn {
namespace helpers {
/**
 * Helper function to provide the rounded up ratio of two integers if the
 * numerator is positive, or zero if the numerator is negative.
 * \param num The numerator for the division operation.
 * \param div The denominator for the division operation.
 * \return Returns the ratio of two integers rounded up if the numerator is
 *         positive, or zero if the numerator is negative.
 */
template <
    typename Index, typename DependentIndexType = Index,
    typename std::enable_if<std::is_same<Index, DependentIndexType>::value &&
                                std::is_signed<DependentIndexType>::value,
                            int>::type = 0>
inline SNN_ALWAYS_INLINE Index
round_ratio_up_above_zero(Index const num, NonDeducedType<Index> const div) {
  static_assert(std::is_integral<Index>::value,
                "round_ratio_up_above_zero is only valid for integral types");
  SNN_ASSERT(div != 0, "Cannot divide by zero");
  return num < 0 ? 0 : (num % div != 0 ? num / div + 1 : num / div);
}

template <
    typename Index, typename DependentIndexType = Index,
    typename std::enable_if<std::is_same<Index, DependentIndexType>::value &&
                                std::is_unsigned<DependentIndexType>::value,
                            int>::type = 0>
inline SNN_ALWAYS_INLINE Index
round_ratio_up_above_zero(Index const num, NonDeducedType<Index> const div) {
  static_assert(std::is_integral<Index>::value,
                "round_ratio_up_above_zero is only valid for integral types");
  SNN_ASSERT(div != 0, "Cannot divide by zero");
  return num % div != 0 ? num / div + 1 : num / div;
}

/**
 * Helper function to provide the ratio of two integers, always rounded up.
 * \param num The numerator of the integer ratio.
 * \param div The divisor of the integer ratio.
 * \return The integral ratio of num and dim, rounded toward +Inf.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE Index round_ratio_up(Index const num,
                                              NonDeducedType<Index> const div) {
  static_assert(std::is_integral<Index>::value,
                "round_ratio_up is only valid for integral types");
  SNN_ASSERT(div != 0, "Cannot divide by zero");
  Index quotient = num / div;
  Index additive = num % div == 0 ? 0 : 1;
  return num < 0 ? quotient : quotient + additive;
}

/**
 * Helper function to round up an integral value to the nearest multiple of a
 * given multiplier.
 *
 * NOTE: This is not implemented for negative integers, and will provide
 * incorrect results if used with them.
 * \param val The value to round up to the nearest positive multiple of
 *            multiplier.
 * \param multiplier The value to round val to a multiple of.
 * \return Returns val rounded up to the nearest positive multiple of
 *         multiplier.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE Index round_up_to_nearest_multiple(
    Index val, NonDeducedType<Index> const multiplier) {
  static_assert(
      std::is_integral<Index>::value,
      "round_up_to_nearest_multiple is only valid for integral types");
  SNN_ASSERT(
      val >= 0,
      "round_up_to_nearest_multiple is not implemented for negative values");
  SNN_ASSERT(multiplier > 0,
             "round_up_to_nearest_multiple is not implemented for negative "
             "multipliers");
  Index const diff = val % multiplier;
  if (diff > 0) {
    val += (multiplier - diff);
  }
  return val;
}
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_HELPERS_RATIO_H_
