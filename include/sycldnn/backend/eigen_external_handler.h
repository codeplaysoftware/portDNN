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

#include <utility>

namespace sycldnn {
namespace backend {
/**
 * Handler to provide access to buffers from externally passed Eigen pointers.
 */
struct EigenExternalHandler {
  template <typename T>
  using pointer_type = T*;

  EigenExternalHandler(Eigen::SyclDevice const& device) : device_(device) {}

  // This deduced return type is required to ensure that the buffer type
  // matches the allocator used in the Eigen device. We cannot assume that
  // std::allocator is used.
  template <typename T>
  auto get_buffer(T* ptr, size_t /*n_elems*/) -> decltype(
      std::declval<Eigen::SyclDevice>()
          .get_sycl_buffer(ptr)
          .template reinterpret<T>(std::declval<cl::sycl::range<1>>())) {
    auto raw_buffer = device_.get_sycl_buffer(ptr);
    auto buffer_size = raw_buffer.get_size();
    assert(buffer_size % sizeof(T) == 0);
    auto cast_size = cl::sycl::range<1>{buffer_size / sizeof(T)};
    return raw_buffer.template reinterpret<T>(cast_size);
  }
  template <typename T>
  size_t get_offset(T* ptr) {
    return device_.get_offset(ptr) / sizeof(T);
  }
  cl::sycl::queue get_queue() { return device_.sycl_queue(); }

 private:
  Eigen::SyclDevice const& device_;
};
}  // namespace backend
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_BACKEND_EIGEN_BACKEND_H_
