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
#ifndef PORTDNN_INCLUDE_BACKEND_EIGEN_INTERNAL_HANDLER_H_
#define PORTDNN_INCLUDE_BACKEND_EIGEN_INTERNAL_HANDLER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenInternalHandler,
 * which provides single and batch matrix multiply implementations using Eigen,
 * as well as internal tensor allocation and buffer fetching methods.
 */
#include <utility>

#include "portdnn/backend/backend_traits.h"
#include "portdnn/backend/crtp_backend.h"
#include "portdnn/helpers/macros.h"

namespace sycldnn {
namespace backend {
/**
 * Handler struct to provide matmul and batch_matmul implementations using
 * Eigen, as well as internal tensor allocations and buffer fetching methods.
 *
 * This expects the Eigen Tensor module to have already been included. We don't
 * explicitly include it in this file so that the user has control of how Eigen
 * is included and which files are actually needed.
 */
template <typename EigenBackend>
struct EigenInternalHandler
    : public CRTPBackend<EigenBackend, EigenInternalHandler> {
 private:
  /** The pointer representation required by the internal handler. */
  template <typename T>
  using internal_pointer_type =
      typename BackendTraits<EigenBackend>::template internal_pointer_type<T>;

 public:
  /**
   * Allocate a tensor to be used internally.
   * \param n_bytes The size of the allocation in bytes.
   * \return Returns a pointer to allocation, using the internal pointer
   *         representation.
   * */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_bytes) {
    auto eigen_device = this->underlying_backend().get_eigen_device();
    return static_cast<internal_pointer_type<T>>(
        eigen_device.allocate(n_bytes));
  }

  /**
   * Deallocate an internal tensor.
   * \param ptr A pointer to the allocation to deallocate.
   */
  template <typename T>
  void deallocate(internal_pointer_type<T> ptr) {
    auto eigen_device = this->underlying_backend().get_eigen_device();
    eigen_device.deallocate(ptr);
  }

  /**
   * Get a MemObject containing the buffer corresponding to a given pointer.
   * \param ptr     A pointer referring to a SYCL buffer with some offset.
   * \param n_elems The number of elements required within the MemObject.
   * \return Returns a MemObject corresponding to the pointer.
   */
  template <typename T>
  auto get_mem_object_internal(internal_pointer_type<T> ptr, size_t n_elems)
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
#endif  // PORTDNN_INCLUDE_BACKEND_EIGEN_MATMUL_HANDLER_H_
