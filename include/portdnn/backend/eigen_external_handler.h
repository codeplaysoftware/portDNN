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
#ifndef PORTDNN_INCLUDE_BACKEND_EIGEN_EXTERNAL_HANDLER_H_
#define PORTDNN_INCLUDE_BACKEND_EIGEN_EXTERNAL_HANDLER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenExternalHandler,
 * which provides access to buffers from externally passed Eigen pointers.
 */

#include <utility>

#include "portdnn/mem_object.h"

#include "portdnn/backend/backend_traits.h"
#include "portdnn/backend/crtp_backend.h"

#include "portdnn/helpers/macros.h"

namespace sycldnn {
namespace backend {

/**
 * Handler to provide access to buffers from externally passed Eigen pointers.
 */
template <typename EigenBackend>
struct EigenExternalHandler
    : public CRTPBackend<EigenBackend, EigenExternalHandler> {
 private:
  /** The pointer representation required by the external handler. */
  template <typename T>
  using pointer_type =
      typename BackendTraits<EigenBackend>::template pointer_type<T>;

 public:
  /**
   * Get a MemObject containing the buffer corresponding to a given pointer.
   * \param ptr     A pointer referring to a SYCL buffer with some offset.
   * \param n_elems The number of elements required within the MemObject.
   * \return Returns a MemObject corresponding to the pointer.
   */
  template <typename T>
  auto get_mem_object(pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_mem_object(
          std::declval<Eigen::SyclDevice>()
              .get_sycl_buffer(ptr)
              .template reinterpret<T>(std::declval<cl::sycl::range<1>>()),
          n_elems, 0u)) {
    // This deduced return type is required to ensure that the allocator type
    // in the returned MemObject matches the allocator used in the Eigen device.
    // We cannot assume that std::allocator is used.
    auto eigen_device = this->underlying_backend().get_eigen_device();
    auto raw_buffer = eigen_device.get_sycl_buffer(ptr);
    auto buffer_size = raw_buffer.byte_size();
    SNN_ASSERT(buffer_size % sizeof(T) == 0,
               "Buffer size must exactly divide the size of its type.");
    auto cast_size = buffer_size / sizeof(T);
    SNN_ASSERT(cast_size >= n_elems,
               "Buffer must contain at least n_elems elements.");
    auto cast_range = cl::sycl::range<1>{cast_size};
    auto typed_buffer = raw_buffer.template reinterpret<T>(cast_range);
    auto offset = eigen_device.get_offset(ptr) / sizeof(T);
    return make_mem_object(typed_buffer, n_elems, offset);
  }
};
}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_EIGEN_EXTERNAL_HANDLER_H_
