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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_INTERNAL_POINTER_SET_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_INTERNAL_POINTER_SET_H_

#include "portdnn/internal/helpers/internal_pointer.h"

namespace sycldnn {
namespace conv2d {
namespace internal {

/**
 * Set of internal pointers constructed from external pointers.
 *
 * The internal pointers will be released through the backend on destruction.
 */
template <typename T, typename Backend>
struct InternalPointerSet {
  using ConstExternalPointer = typename Backend::template pointer_type<T const>;
  using ExternalPointer = typename Backend::template pointer_type<T>;
  using ConstInternalPointer =
      ::sycldnn::internal::helpers::InternalPointer<T const, Backend>;
  using InternalPointer =
      ::sycldnn::internal::helpers::InternalPointer<T, Backend>;

  InternalPointerSet(ConstExternalPointer input, ConstExternalPointer filter,
                     ExternalPointer output, Backend& backend)
      : input{input, backend},
        filter{filter, backend},
        output{output, backend} {}

  ConstInternalPointer input;
  ConstInternalPointer filter;
  InternalPointer output;
};

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_INTERNAL_POINTER_SET_H_
