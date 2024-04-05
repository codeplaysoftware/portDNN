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
#ifndef SYCLDNN_INCLUDE_REDUCE_INTERNAL_IDENTITIES_H_
#define SYCLDNN_INCLUDE_REDUCE_INTERNAL_IDENTITIES_H_

#include <numeric>

#include "sycldnn/reduce/operators.h"

namespace sycldnn {
namespace reduce {
namespace internal {

/** Static constexpr member `value` is set to the additive identity 0 */
template <typename T, typename Op>
struct Identity {
  static T constexpr value = 0.f;
};

/** Static constexpr member `value` is set to the lowest value of type T */
template <typename T>
struct Identity<T, Max> {
  static T constexpr value = std::numeric_limits<T>::lowest();
};
/** Static constexpr member `value` is set to the highest value of type T */
template <typename T>
struct Identity<T, Min> {
  static T constexpr value = std::numeric_limits<T>::max();
};

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_REDUCE_INTERNAL_IDENTITIES_H_
