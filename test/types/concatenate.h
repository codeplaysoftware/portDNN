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
#ifndef PORTDNN_TEST_TYPES_CONCATENATE_H_
#define PORTDNN_TEST_TYPES_CONCATENATE_H_

#include <gtest/gtest.h>

#include "test/types/type_list.h"

#include <type_traits>

namespace sycldnn {
namespace types {

template <typename... Ts>
struct Concatenate;
/** Concatenate two type lists together into one type list. */
template <typename... Ts, typename... Us>
struct Concatenate<TypeList<Ts...>, TypeList<Us...>> {
  using type = TypeList<Ts..., Us...>;
};
/**
 * The googletest Types<...> list implementation is hardcoded to always contain
 * the same number of elements (with any unspecified types being `None`). This
 * means that concatenate does not work as expected with these types.
 *
 * Instead use sycldnn::types::TypeList<...> and use ToGTestTypes<...>::type
 * before passing to googletest.
 */
template <typename T, typename... Ts, typename... Us>
struct Concatenate<::testing::Types<T, Ts...>, ::testing::Types<Us...>> {
 private:
  struct not_implemented;
  static_assert(std::is_same<not_implemented, T>::value,
                "Concatenate is not implemented for googletest "
                "::testing::Types<...>, use sycldnn::types::TypeList<...> "
                "instead");
};

static_assert(std::is_same<TypeList<char, int, unsigned, float, double>,
                           Concatenate<TypeList<char, int, unsigned>,
                                       TypeList<float, double>>::type>::value,
              "Error when concatenating two type lists");

/** Concatenate any number of type lists. */
template <typename... Ts, typename... Us, typename... Lists>
struct Concatenate<TypeList<Ts...>, TypeList<Us...>, Lists...> {
  using type = typename Concatenate<TypeList<Ts..., Us...>, Lists...>::type;
};
static_assert(
    std::is_same<TypeList<char, int, unsigned, float, double>,
                 Concatenate<TypeList<char, int>, TypeList<unsigned>,
                             TypeList<float>, TypeList<double>>::type>::value,
    "Error when concatenating multiple type lists");

}  // namespace types
}  // namespace sycldnn
#endif  // PORTDNN_TEST_TYPES_CONCATENATE_H_
