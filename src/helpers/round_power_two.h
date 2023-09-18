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
#ifndef PORTDNN_SRC_HELPERS_ROUND_POWER_TWO_H_
#define PORTDNN_SRC_HELPERS_ROUND_POWER_TWO_H_

#include "portdnn/helpers/macros.h"

#include <cmath>

namespace sycldnn {
namespace helpers {
/**
 * Round a value up to the nearest power of two which is greater or equal to the
 * value.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE Index round_to_power_of_two(Index value) {
  auto log_value = std::ceil(std::log2(value));
  return static_cast<Index>(std::exp2(log_value));
}
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_SRC_HELPERS_ROUND_POWER_TWO_H_
