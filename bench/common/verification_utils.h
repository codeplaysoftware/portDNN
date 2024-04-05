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
#ifndef SYCLDNN_BENCH_COMMON_VERIFICATION_UTILS_H_
#define SYCLDNN_BENCH_COMMON_VERIFICATION_UTILS_H_

#include <algorithm>
#include <iostream>
#include <vector>

namespace sycldnn {
namespace bench {
namespace internal {

/**
 * Compare expected results to predicted results to ensure it's within the
 * specified tolerance level.
 *
 * Prints to stderr if there is an absolute difference
 * greater than the tolerance and return false, otherwise returns true.
 * TODO add better way of comparing fp results, similar to the ULPS
 * comparisson in the testing framework.
 */
template <typename DataType>
bool compare_results(const std::vector<DataType>& expected,
                     const std::vector<DataType>& predicted,
                     DataType tolerance) {
  assert(expected.size() == predicted.size() &&
         "Predicted and expected vectors must be of the same size");
  for (int i = 0; i < (int)expected.size(); i++) {
    auto absdiff = std::abs(expected[i] - predicted[i]);
    if (absdiff > tolerance) {
      fprintf(
          stderr,
          "The absolute difference %f is greater than the given tolerance %f "
          "at element %d between the expected result %f and the predicted "
          "result %f\n",
          absdiff, tolerance, i, expected[i], predicted[i]);
      return false;
    }
  }
  return true;
}

/**
 * Fill a vector with `value, value+1,...` with `size` elements.
 *
 * This has the same behaviour as `std::iota`, but will append `size` elements
 * to the back of the provided vector, rather than overwriting the existing
 * values.
 */
template <typename DataType>
void iota_n(std::vector<DataType>& c, size_t size, DataType value) {
  c.reserve(size);
  std::generate_n(std::back_inserter(c), size, [&value] { return value++; });
}

/**
 * Fill a vector with the values:
 *   `init_value, init_value+1, ..., max_value-1, max_value, init_value,...`
 * where the values will increase by `1` each step, but the values are
 * limited by `max_value`. Once `max_value` is reached, the values begin
 * again at init_value.
 */
template <typename DataType>
void iota_n_modulo(std::vector<DataType>& c, size_t size, DataType init_value,
                   DataType max_value) {
  if (max_value < static_cast<DataType>(1)) {
    return iota_n(c, size, init_value);
  }
  c.reserve(size);
  size_t n_steps = static_cast<size_t>(max_value - init_value) + 1;
  int counter = 0;
  std::generate_n(std::back_inserter(c), size, [&counter, n_steps, init_value] {
    return init_value + (counter++ % n_steps);
  });
}

/**
 * Get a vector of the required size initialised as with iota_n_modulo.
 *
 * The vector returned will contain `size` elements of the values:
 *   `1, 2, ..., max_value-1, max_value, 1,...`
 *
 * If the `max_val` passed to this function is less than 1 then the maximum
 * value will be treated as `size`, that is the values will be:
 *   `1, 2, ..., size-1, size`
 */
template <typename DataType>
std::vector<DataType> iota_initialised_data(size_t size, DataType max_val) {
  std::vector<DataType> data;
  internal::iota_n_modulo(data, size, static_cast<DataType>(1), max_val);
  return data;
}

/**
 * Get a vector of the required size initialised as with
 * iota_initialised_data.
 *
 * The vector returned will contain `size` elements of the values:
 *   `-n, -n+1, ..., 0, 1, ..., n-2, n-1`
 * where n is half of `size`, rounded up when `size` is odd.
 */
template <typename DataType>
std::vector<DataType> iota_initialised_signed_data(size_t size) {
  std::vector<DataType> data;
  DataType difference = (size % 2 == 0) ? (size / 2) : ((size + 1) / 2);
  DataType min = -difference;
  DataType max = size - 1 - difference;
  internal::iota_n_modulo(data, size, min, max);
  return data;
}
}  // namespace internal
}  // namespace bench
}  // namespace sycldnn
#endif
