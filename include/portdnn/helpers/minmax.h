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
#ifndef PORTDNN_INCLUDE_HELPERS_MINMAX_H_
#define PORTDNN_INCLUDE_HELPERS_MINMAX_H_

/**
 * \file
 * Implements the \ref sycldnn::helpers::min() and \ref sycldnn::helpers::max()
 * functions. The functions provide the compiler improved visibility for
 * optimization purposes relative to the SYCL builtin functions.
 */
#include "portdnn/helpers/macros.h"

#include <type_traits>

namespace sycldnn {
namespace helpers {

/**
 * Min function. Prefer this over cl::sycl::min to allow the compiler to
 * understand more about the code.
 * \param a The first operand.
 * \param b The second operand.
 * \return Returns the minimum value of operands a and b.
 */
template <typename T, typename U>
inline SNN_ALWAYS_INLINE typename std::common_type<T, U>::type min(T a, U b) {
  return (a < b) ? a : b;
}

/**
 * Max function. Prefer this over cl::sycl::max to allow the compiler to
 * understand more about the code.
 * \param a The first operand.
 * \param b The second operand.
 * \return Returns the maximum value of operands a and b.
 */
template <typename T, typename U>
inline SNN_ALWAYS_INLINE typename std::common_type<T, U>::type max(T a, U b) {
  return (a > b) ? a : b;
}
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_HELPERS_MINMAX_H_
