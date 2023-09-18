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
#ifndef PORTDNN_TEST_TYPES_TYPE_LIST_H_
#define PORTDNN_TEST_TYPES_TYPE_LIST_H_

namespace sycldnn {
namespace types {

/** Variadic list of types. */
template <typename... Ts>
struct TypeList {};

}  // namespace types
}  // namespace sycldnn
#endif  // PORTDNN_TEST_TYPES_TYPE_LIST_H_
