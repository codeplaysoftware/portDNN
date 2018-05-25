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

namespace sycldnn {
namespace backend {
/**
 * Handler to provide access to buffers from externally passed Eigen pointers.
 */
struct EigenExternalHandler {
  /** The pointer representation required by the external handler. */
  template <typename T>
  using pointer_type = T*;

  /**
   * Constructs an instance of \ref sycldnn::backend::EigenExternalHandler from
   * an instance of Eigen's SyclDevice.
   * \param device The Eigen::SyclDevice to construct the handler from.
   */
  EigenExternalHandler(Eigen::SyclDevice const& device) : device_(device) {}

  /**
   * Get a SYCL buffer from an external pointer.
   * \param ptr The pointer for which to retrieve the corresponding SYCL buffer.
   * \return Returns a SYCL buffer corresponding to ptr.
   */
  template <typename T>
  auto get_buffer(pointer_type<T> ptr, size_t /*n_elems*/) -> decltype(
      std::declval<Eigen::SyclDevice>()
          .get_sycl_buffer(ptr)
          .template reinterpret<T>(std::declval<cl::sycl::range<1>>())) {
    // This deduced return type is required to ensure that the buffer type
    // matches the allocator used in the Eigen device. We cannot assume that
    // std::allocator is used.
    auto raw_buffer = device_.get_sycl_buffer(ptr);
    auto buffer_size = raw_buffer.get_size();
    assert(buffer_size % sizeof(T) == 0);
    auto cast_size = cl::sycl::range<1>{buffer_size / sizeof(T)};
    return raw_buffer.template reinterpret<T>(cast_size);
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
    return device_.get_offset(ptr) / sizeof(T);
  }

  /**
   * Gets the SYCL queue that the backend is bound to.
   * \return Returns the SYCL queue that the backend is bound to.
   */
  cl::sycl::queue get_queue() { return device_.sycl_queue(); }

 private:
  Eigen::SyclDevice const& device_;
};
}  // namespace backend
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_BACKEND_EIGEN_BACKEND_H_
