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
#ifndef PORTDNN_INCLUDE_INTERNAL_HELPERS_INTERNAL_POINTER_H_
#define PORTDNN_INCLUDE_INTERNAL_HELPERS_INTERNAL_POINTER_H_

#include "portdnn/helpers/macros.h"

namespace sycldnn {
namespace internal {
namespace helpers {

/** Helper to convert an external pointer to an internal pointer. */
template <typename T, typename Backend>
struct InternalPointer {
  using ExternalPointer = typename Backend::template pointer_type<T>;
  using Pointer = typename Backend::template internal_pointer_type<T>;

  /**
   * Convert the given pointer to an internal pointer using the backend.
   *
   * \param ptr     External pointer to convert
   * \param backend Backend to use to convert the pointer
   */
  InternalPointer(ExternalPointer const& ptr, Backend& backend)
      : pointer{backend.to_internal_pointer(ptr)}, backend{backend} {}

  ~InternalPointer() { backend.release_internal_pointer(pointer); }

  /** Get the underlying pointer type. */
  Pointer get() const { return pointer; }

  Pointer pointer;

 private:
  Backend& backend;
  SNN_DISABLE_COPY(InternalPointer);
  SNN_DISABLE_MOVE(InternalPointer);
};

}  // namespace helpers
}  // namespace internal
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_HELPERS_INTERNAL_POINTER_H_
