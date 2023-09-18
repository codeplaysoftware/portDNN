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
#ifndef PORTDNN_INCLUDE_HELPERS_DIMS_H_
#define PORTDNN_INCLUDE_HELPERS_DIMS_H_

/**
 * \file
 * Contains helper functions related to vector dimensions.
 */

#include <functional>
#include <numeric>

namespace sycldnn {
namespace helpers {

/**
 * @brief Compute the total size of \p dims
 *
 * @tparam Array compatible with std::begin and std::end
 * @param dims Array of dimensions
 * @return Total size of \p dims
 */
template <class Array>
inline size_t get_total_size(const Array& dims) {
  return std::accumulate(std::begin(dims), std::end(dims), 1,
                         std::multiplies<>());
}

}  // namespace helpers
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_HELPERS_DIMS_H_
