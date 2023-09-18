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
#ifndef PORTDNN_TEST_TYPES_NESTED_PAIRS_TO_TRIPLE_H_
#define PORTDNN_TEST_TYPES_NESTED_PAIRS_TO_TRIPLE_H_

#include <gtest/gtest.h>

#include "test/types/type_list.h"
#include "test/types/type_pair.h"
#include "test/types/type_triple.h"

#include <type_traits>

namespace sycldnn {
namespace types {

template <typename T>
struct NestedPairsToTriple {
 private:
  struct not_implemented;
  static_assert(std::is_same<not_implemented, T>::value,
                "NestedPairsToTriple is only implemented for nested TypePairs");
};

template <typename... Ts>
struct NestedPairsToTriple<::testing::Types<Ts...>> {
 private:
  struct not_implemented;
  static_assert(
      std::is_same<not_implemented, Ts...>::value,
      "NestedPairsToTriple is not implemented for a googletest type list"
      "::testing::Types<...>, use sycldnn::types::TypeList<...> instead");
};

template <typename T, typename U, typename V>
struct NestedPairsToTriple<TypePair<T, TypePair<U, V>>> {
  using type = TypeTriple<T, U, V>;
};

template <typename T, typename U, typename V>
struct NestedPairsToTriple<TypePair<TypePair<T, U>, V>> {
  using type = TypeTriple<T, U, V>;
};

static_assert(
    std::is_same<
        NestedPairsToTriple<TypePair<TypePair<int, float>, double>>::type,
        TypeTriple<int, float, double>>::value,
    "Error in converting Pair<Pair<T, U>, V> to Triple<T, U, V>");

static_assert(
    std::is_same<
        NestedPairsToTriple<TypePair<int, TypePair<float, double>>>::type,
        TypeTriple<int, float, double>>::value,
    "Error in converting Pair<T, Pair<U, V>> to Triple<T, U, V>");

template <typename... Ts>
struct NestedPairsToTriple<TypeList<Ts...>> {
  using type = TypeList<typename NestedPairsToTriple<Ts>::type...>;
};

static_assert(
    std::is_same<NestedPairsToTriple<
                     TypeList<TypePair<int, TypePair<float, double>>,
                              TypePair<double, TypePair<float, int>>>>::type,
                 TypeList<TypeTriple<int, float, double>,
                          TypeTriple<double, float, int>>>::value,
    "Error in converting a list of nested pairs to a list of triples");

}  // namespace types
}  // namespace sycldnn

#endif  // PORTDNN_TEST_TYPES_NESTED_PAIRS_TO_TRIPLE_H_
