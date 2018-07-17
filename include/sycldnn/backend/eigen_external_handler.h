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
#ifndef SYCLDNN_INCLUDE_BACKEND_EIGEN_EXTERNAL_HANDLER_H_
#define SYCLDNN_INCLUDE_BACKEND_EIGEN_EXTERNAL_HANDLER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::EigenExternalHandler,
 * which provides access to buffers from externally passed Eigen pointers.
 */

#include <utility>

#include "sycldnn/backend/backend_traits.h"
#include "sycldnn/backend/crtp_backend.h"

#include "sycldnn/helpers/macros.h"

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
   * Get a SYCL buffer from an external pointer.
   * \param ptr The pointer for which to retrieve the corresponding SYCL buffer.
   * \param n_elems The number of elements in the buffer.
   * \return Returns a SYCL buffer corresponding to ptr.
   */
  template <typename T>
  auto get_buffer(pointer_type<T> ptr, size_t n_elems) -> decltype(
      std::declval<Eigen::SyclDevice>()
          .get_sycl_buffer(ptr)
          .template reinterpret<T>(std::declval<cl::sycl::range<1>>())) {
    // This deduced return type is required to ensure that the buffer type
    // matches the allocator used in the Eigen device. We cannot assume that
    // std::allocator is used.
    auto eigen_device = this->underlying_backend().get_eigen_device();
    auto raw_buffer = eigen_device.get_sycl_buffer(ptr);
    auto buffer_size = raw_buffer.get_size();
    SNN_ASSERT(buffer_size % sizeof(T) == 0,
               "Buffer size must exactly divide the size of its type.");
    auto cast_size = buffer_size / sizeof(T);
    SNN_ASSERT(cast_size >= n_elems,
               "Buffer must contain at least n_elems elements.");
    // n_elems is used for the debug assert, but otherwise unused.
    SNN_UNUSED_VAR(n_elems);
    auto cast_range = cl::sycl::range<1>{cast_size};
    return raw_buffer.template reinterpret<T>(cast_range);
  }

  /**
   * Get the offset from an external pointer. An external pointer may be an
   * offset from some base address, where the base address corresponds to a
   * SYCL buffer, and the offset refers to some address internal to the SYCL
   * buffer. This function enables querying such an offset.
   * \param ptr The external pointer to query the offset for.
   * \return Returns the offset from the buffer base address, in elements.
   */
  template <typename T>
  size_t get_offset(pointer_type<T> ptr) {
    auto eigen_device = this->underlying_backend().get_eigen_device();
    return eigen_device.get_offset(ptr) / sizeof(T);
  }
};
}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_EIGEN_EXTERNAL_HANDLER_H_
