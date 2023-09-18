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
#ifndef PORTDNN_INCLUDE_HELPERS_NON_DEDUCED_TYPE_H_
#define PORTDNN_INCLUDE_HELPERS_NON_DEDUCED_TYPE_H_

/**
 * \file
 * Provides sycldnn::helpers::NonDeduced type and
 * sycldnn::helpers::NonDeducedType alias.
 */

namespace sycldnn {
namespace helpers {

/**
 * Helper struct to provide a type in a non deduced context.
 *
 * Many functions in portDNN are templated, with most of the function
 * parameters being used as the same type. However there is rarely a requirement
 * that all the types are exactly the same, only that one type can be converted
 * to another. e.g. in the following it would be valid to divide a `long` by an
 * `int`, as the `int` can be easily be extended to `long`.
 *
 * \code
 * template <typename Index>
 * Index round_ratio_up(Index num, Index div);
 * \endcode
 *
 * Rather than having function templates that have a different template
 * parameter for each function parameter (and hence a new function declaration
 * for all pairs of types) we can ensure that the single template parameter is
 * used in a non-deducible context in the function parameters. This helps the
 * compiler choose the right function instantiation and will automatically
 * convert the function parameters.
 *
 * \code
 * template <typename Index>
 * Index round_ratio_up(Index num, NonDeduced<Index>::type div);
 * \endcode
 *
 * \tparam T The type to provide in a non-deducible context.
 */
template <typename T>
struct NonDeduced {
  /**
   * The class template type T, but in a non-deducible context.
   */
  using type = T;
};

/**
 * Helper alias to provide NonDeduced<T>::type.
 *
 * \see \ref sycldnn::helpers::NonDeduced
 */
template <typename T>
using NonDeducedType = typename NonDeduced<T>::type;

}  // namespace helpers
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_HELPERS_NON_DEDUCED_TYPE_H_
