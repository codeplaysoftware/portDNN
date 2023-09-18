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
#ifndef PORTDNN_SRC_HELPERS_FAST_DIV_H_
#define PORTDNN_SRC_HELPERS_FAST_DIV_H_

#include <CL/sycl.hpp>

#include "portdnn/helpers/macros.h"

namespace sycldnn {
namespace fast_div {
/**
 * This uses the fast integer division technique outlined in "Division by
 * Invariant Integers using Multiplication" by Granlund and Montgomery
 * (http://dx.doi.org/10.1145/773473.178249), and the implementation is based
 * on that found in Chapter 10 (Figure 10-1) of "Hackers Delight" by Warren.
 *
 * The idea behind this fast division algorithm is to perform some additional
 * computations on the host to compute suitable magic numbers to convert each
 * division on the device into a multiply followed by a shift.
 *
 * The key component to this is the mul_hi operation, which takes two integers
 * and multiplies them using twice the number of bits before returning the top
 * half of the bits. In the 32 bit case, this is equivalent to performing a 64
 * bit multiply and shifting the result left by 32. Mathematically this is
 * equivalent to:
 *
 *   mul_hi(x, y) = floor(x * y / 2^32)
 *
 * If the mul_hi operation is followed by a shift left by 'z' bits, then the
 * whole fast division is equivalent to:
 *
 *     fast_div(x, y, z) = mul_hi(x, y) >> z = floor(mul_hi(x, y) / 2^z) =
 *     floor( floor(x * y / 2^32) / 2^z) = floor( x * y / 2^(32 + z) )
 *
 * More generally, for W-bit integers, for a given divisor 'd', we need the
 * smallest multiple 'm' and shift 's' satisfying:
 *
 *     floor(m * n / 2^(W + s)) = floor(n / d)
 *
 * for every possible signed integer 'n' where 0 <= n < 2^(W-1).
 *
 * The smallest such multiple can be any integer between 0 and 2^W, however the
 * largest representable integer in the signed integer is 2^(W-1), so the
 * multiple must be stored in an unsigned integer and the mul_hi operation must
 * also be computed using unsigned types.
 *
 * Let 'p = W + s', then we need 'm' to be the next integer greater than '2^p /
 * d', that is
 *
 * (1)  m = (2^p + d - (2^p % d) ) / d
 *
 * We can find 'p' by using the largest representable integer 'nc' such that
 * (nc % d) = d - 1, or equivalently
 *
 *     nc = 2^(W-1) - (2^(W-1) % d) - 1
 *
 * Then p can be found using the inequality:
 *
 * (2)  2^p > nc * ( d - (2^p, d) )
 *
 * and the fact that if 'p_0' satisfies this, then so does 'p_0 + 1'.
 *
 * We know 'p' is at least W, so starting with this we can try each value of
 * 'p' until we find the smallest value satisfying (2). This will give the
 * shift value 's = p - W', and (1) will give the value for m.
 *
 * In this implementation we assume that the divisor is positive, which allows
 * us to skip certain branches and checks otherwise required. This approach
 * also only works for divisors strictly greater than 1.
 */
template <typename Index>
struct FastDiv {
  static_assert(std::is_signed<Index>::value,
                "Index type for fast division must be a signed type.");
  using Unsigned = typename std::make_unsigned<Index>::type;
  /**
   * The FastDiv constructor calculates the required magic numbers for
   * converting the division to a multiply and shift. As this constructor is not
   * a trivial computation, this should only be used on the host and then the
   * values computed on the host can be passed as parameters to the SYCL kernel.
   */
  explicit FastDiv(Index const divisor) {
    SNN_ASSERT(divisor > 1,
               "FastDiv requires the divisor to be greater than 1");

    int constexpr index_bit_length = std::numeric_limits<Index>::digits;
    Unsigned constexpr two_pow = static_cast<Unsigned>(1) << index_bit_length;

    auto const unsigned_d = static_cast<Unsigned>(divisor);
    Unsigned const nc = two_pow - 1 - (two_pow % unsigned_d);

    int power = index_bit_length;
    Unsigned two_p_quot_nc = two_pow / nc;
    Unsigned two_p_rem_nc = two_pow % nc;
    Unsigned two_p_quot_d = two_pow / unsigned_d;
    Unsigned two_p_rem_d = two_pow % unsigned_d;

    auto increase_two_power_by_one = [](Unsigned div, Unsigned& quot,
                                        Unsigned& rem) {
      quot *= 2;
      rem *= 2;
      if (rem >= div) {
        ++quot;
        rem -= div;
      }
    };

    Unsigned delta;
    do {
      ++power;
      increase_two_power_by_one(nc, two_p_quot_nc, two_p_rem_nc);
      increase_two_power_by_one(unsigned_d, two_p_quot_d, two_p_rem_d);

      delta = unsigned_d - two_p_rem_d;

    } while (two_p_quot_nc < delta ||
             (two_p_quot_nc == delta && two_p_rem_nc == 0));

    multiple = two_p_quot_d + 1;
    shift = power - index_bit_length - 1;
  }

  /**
   * Perform the actual division using the FastDiv magic numbers.
   */
  SNN_ALWAYS_INLINE Index divide(Index value) {
    SNN_ASSERT(value >= 0, "FastDiv requires nonnegative values");
    auto unsigned_value = static_cast<Unsigned>(value);
    Unsigned unsigned_ans = cl::sycl::mul_hi(unsigned_value, multiple) >> shift;
    return static_cast<Index>(unsigned_ans);
  }

  Unsigned multiple;
  Index shift;
};
/**
 * Operator overload to allow FastDiv to be used in the same way as an index
 * value.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE Index operator/(Index value, FastDiv<Index> fast_div) {
  static_assert(std::is_signed<Index>::value,
                "Fast division is only supported on signed integer types.");
  return fast_div.divide(value);
}
/**
 * Helper struct to choose whether to use the FastDiv construct or not.
 */
template <typename Index, bool UseFastDiv>
struct IndexDiv {
  using type = Index;
};
template <typename Index>
struct IndexDiv<Index, true> {
  using type = FastDiv<Index>;
};
}  // namespace fast_div
}  // namespace sycldnn
#endif  // PORTDNN_SRC_HELPERS_FAST_DIV_H_
