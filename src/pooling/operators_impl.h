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

#ifndef PORTDNN_SRC_POOLING_OPERATORS_IMPL_H_
#define PORTDNN_SRC_POOLING_OPERATORS_IMPL_H_

#include <CL/sycl.hpp>

#include "portdnn/pooling/operators.h"

namespace sycldnn {
namespace pooling {

template <typename T>
struct Max {
  T max;
  Max() : max(std::numeric_limits<T>::lowest()) {}
  void accumulate(T val) { val > max ? max = val : T(0); }
  T value() { return max; }
};

template <typename T, int N>
struct Max<cl::sycl::vec<T, N>> {
  cl::sycl::vec<T, N> max;
  Max() : max(std::numeric_limits<T>::lowest()) {}
  void accumulate(cl::sycl::vec<T, N> val) { max = cl::sycl::max(max, val); }
  cl::sycl::vec<T, N> value() { return max; }
};

template <typename T>
struct MaxWithNan {
  T max = std::numeric_limits<T>::lowest();

  void accumulate(T val) {
    if (cl::sycl::isnan(val) || val > max) {
      max = val;
    }
  }
  T value() { return max; }
};

template <typename T, int N>
struct MaxWithNan<cl::sycl::vec<T, N>> {
  using VecType = cl::sycl::vec<T, N>;
  VecType max = VecType{std::numeric_limits<T>::lowest()};

  void accumulate(VecType val) {
    auto select_mask = val != val || val > max;
    max = cl::sycl::select(max, val, select_mask);
  }
  VecType value() { return max; }
};

/** Template that will average a sequence of accumulated values. */
template <typename T>
struct Average {
  /** The number of values accumulated. */
  int tally;
  /** The sum of the accumulated values. */
  T sum;

  Average() : tally(0), sum(0) {}

  /** Increases the running total of the struct's accumulator.
   * \param val The next value to be added to the accumulator. */
  void accumulate(T val) {
    tally++;
    sum += val;
  }

  /** Observes the average, by dividing the sum  by the number of tallies.
   * \return The average of all accumulated values. */
  T value() { return sum / T(tally); }
};

template <template <typename> class Op>
struct EqualCheck;

template <>
struct EqualCheck<Max> {
  /** Consider two values equal if they are not NaN and have the same value. */
  template <typename T>
  static bool are_equal(T a, T b) {
    return a == b;
  }
};

template <>
struct EqualCheck<MaxWithNan> {
  /** Consider two values equal if both are NaN or have the same value. */
  template <typename T>
  static bool are_equal(T a, T b) {
    return a == b || (cl::sycl::isnan(a) && cl::sycl::isnan(b));
  }
};

}  // namespace pooling
}  // namespace sycldnn

#endif  // PORTDNN_SRC_POOLING_OPERATORS_IMPL_H_
