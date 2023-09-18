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
#ifndef PORTDNN_TEST_TYPES_NESTED_PAIRS_TO_TUPLE4_H_
#define PORTDNN_TEST_TYPES_NESTED_PAIRS_TO_TUPLE4_H_

#include <gtest/gtest.h>

#include "test/types/type_list.h"
#include "test/types/type_pair.h"
#include "test/types/type_tuple4.h"

#include <type_traits>

namespace sycldnn {
namespace types {

template <typename T>
struct NestedPairsToTuple4 {
 private:
  struct not_implemented;
  static_assert(std::is_same<not_implemented, T>::value,
                "NestedPairsToTuple4 is only implemented for nested TypePairs");
};

template <typename... Ts>
struct NestedPairsToTuple4<::testing::Types<Ts...>> {
 private:
  struct not_implemented;
  static_assert(
      std::is_same<not_implemented, Ts...>::value,
      "NestedPairsToTuple4 is not implemented for a googletest type list"
      "::testing::Types<...>, use sycldnn::types::TypeList<...> instead");
};

template <typename T0, typename T1, typename T2, typename T3>
struct NestedPairsToTuple4<TypePair<T0, TypePair<T1, TypePair<T2, T3>>>> {
  using type = TypeTuple4<T0, T1, T2, T3>;
};

template <typename T0, typename T1, typename T2, typename T3>
struct NestedPairsToTuple4<TypePair<TypePair<TypePair<T0, T1>, T2>, T3>> {
  using type = TypeTuple4<T0, T1, T2, T3>;
};

static_assert(
    std::is_same<NestedPairsToTuple4<TypePair<
                     TypePair<TypePair<char, int>, float>, double>>::type,
                 TypeTuple4<char, int, float, double>>::value,
    "Error in converting Pair<Pair<Pair<T0, T1>, T2>, T3> to Tuple4<T0, T1, "
    "T2, T3>");

static_assert(
    std::is_same<NestedPairsToTuple4<TypePair<
                     char, TypePair<int, TypePair<float, double>>>>::type,
                 TypeTuple4<char, int, float, double>>::value,
    "Error in converting Pair<T0, Pair<T1, Pair<T2, T3>>> to Tuple4<T0, T1, "
    "T2, T3>");

template <typename... Ts>
struct NestedPairsToTuple4<TypeList<Ts...>> {
  using type = TypeList<typename NestedPairsToTuple4<Ts>::type...>;
};

static_assert(
    std::is_same<
        NestedPairsToTuple4<TypeList<
            TypePair<char, TypePair<int, TypePair<float, double>>>,
            TypePair<double, TypePair<float, TypePair<int, char>>>>>::type,
        TypeList<TypeTuple4<char, int, float, double>,
                 TypeTuple4<double, float, int, char>>>::value,
    "Error in converting a list of nested pairs to a list of tuple4s");

}  // namespace types
}  // namespace sycldnn

#endif  // PORTDNN_TEST_TYPES_NESTED_PAIRS_TO_TUPLE4_H_
