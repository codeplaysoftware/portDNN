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
#ifndef PORTDNN_SRC_HELPERS_VECTOR_TYPE_H_
#define PORTDNN_SRC_HELPERS_VECTOR_TYPE_H_

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {
/** Vector type for a given data type and vector size. */
template <typename T, int Width>
struct VectorType {
  using type = cl::sycl::vec<T, Width>;
};
/** For vectors of size 1 just use the underlying data type. */
template <typename T>
struct VectorType<T, 1> {
  using type = T;
};
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_SRC_HELPERS_VECTOR_TYPE_H_
