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
#ifndef PORTDNN_TEST_TYPES_CARTESIAN_PRODUCT_H_
#define PORTDNN_TEST_TYPES_CARTESIAN_PRODUCT_H_

#include <gtest/gtest.h>

#include "test/types/concatenate.h"
#include "test/types/type_list.h"
#include "test/types/type_pair.h"

#include <type_traits>

namespace sycldnn {
namespace types {

template <typename T, typename U>
struct CartesianProduct;
/**
 * The googletest Types<...> list implementation is hardcoded to always contain
 * the same number of elements (with any unspecified types being `None`). This
 * means that the product given below does not work as expected with these
 * types.
 *
 * Instead use sycldnn::types::TypeList<...> and use ToGTestTypes<...>::type
 * before passing to googletest.
 */
template <typename T, typename... Ts, typename... Us>
struct CartesianProduct<::testing::Types<T, Ts...>, ::testing::Types<Us...>> {
 private:
  struct not_implemented;
  static_assert(std::is_same<not_implemented, T>::value,
                "CartesianProduct is not implemented for googletest "
                "::testing::Types<...>, use sycldnn::types::TypeList<...> "
                "instead");
};
/**
 * Compute the product between a type list with a single type and a list with
 * any number of types.
 */
template <typename T, typename... Us>
struct CartesianProduct<TypeList<T>, TypeList<Us...>> {
  // Expands into a TypeList of TypePairs with T as the first type and each
  // element of Us in successive pairs.
  using type = TypeList<TypePair<T, Us>...>;
};
static_assert(
    std::is_same<
        TypeList<TypePair<int, float>, TypePair<int, double>>,
        CartesianProduct<TypeList<int>, TypeList<float, double>>::type>::value,
    "Error in CartesianProduct base case");
/**
 * Compute the Cartesian product between two type lists.
 */
template <typename T, typename... Ts, typename... Us>
struct CartesianProduct<TypeList<T, Ts...>, TypeList<Us...>> {
  // The general case of CartesianProduct which will recursively go through the
  // types in the first TypeList and concatenate the TypePairs into the final
  // TypeList.
  using type = typename Concatenate<
      typename CartesianProduct<TypeList<T>, TypeList<Us...>>::type,
      typename CartesianProduct<TypeList<Ts...>, TypeList<Us...>>::type>::type;
};
static_assert(
    std::is_same<TypeList<TypePair<char, float>, TypePair<char, double>,
                          TypePair<int, float>, TypePair<int, double>>,
                 CartesianProduct<TypeList<char, int>,
                                  TypeList<float, double>>::type>::value,
    "Error in general CartesianProduct");
}  // namespace types
}  // namespace sycldnn
#endif  // PORTDNN_TEST_TYPES_CARTESIAN_PRODUCT_H_
