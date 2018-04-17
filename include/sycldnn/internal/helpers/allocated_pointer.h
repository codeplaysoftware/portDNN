/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_INTERNAL_HELPERS_ALLOCATED_POINTER_H_
#define SYCLDNN_INCLUDE_INTERNAL_HELPERS_ALLOCATED_POINTER_H_

#include "sycldnn/helpers/macros.h"

namespace sycldnn {
namespace internal {
namespace helpers {

/** Helper pointer type to automatically allocate and deallocate a pointer. */
template <typename T, typename Backend>
struct AllocatedPointer {
  using Pointer = typename Backend::template internal_pointer_type<T>;

  /**
   * Allocate a SYCL buffer using the provided backend.
   *
   * \param alloc_size Number of bytes to allocate
   * \param backend    Backend to use to allocate the buffer
   */
  AllocatedPointer(size_t alloc_size, Backend& backend)
      : pointer{backend.template allocate<T>(alloc_size)}, backend{backend} {}

  SNN_DISABLE_COPY(AllocatedPointer);
  SNN_DISABLE_MOVE(AllocatedPointer);

  /** Deallocate pointer on destruction. */
  ~AllocatedPointer() { backend.deallocate(pointer); }

  /** Get the underlying pointer type. */
  Pointer get() const { return pointer; }

 private:
  Pointer pointer;
  Backend& backend;
};

}  // namespace helpers
}  // namespace internal
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_HELPERS_ALLOCATED_POINTER_H_
