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
#ifndef SYCLDNN_SRC_HELPERS_MEM_UTILS_H_
#define SYCLDNN_SRC_HELPERS_MEM_UTILS_H_

#include <CL/sycl.hpp>
#include <type_traits>

#include <iostream>

namespace sycldnn {
namespace helpers {

template <typename T, bool IsUSM>
auto alloc(size_t size, cl::sycl::queue& queue) {
  if constexpr (IsUSM) {
    return cl::sycl::malloc_device<T>(size, queue);
  } else {
    return cl::sycl::buffer<T, 1>(cl::sycl::range<1>(size));
  }
}

template <typename T, bool IsUSM>
auto alloc_and_assign(size_t size, const T* values, cl::sycl::queue& queue) {
  if constexpr (IsUSM) {
    auto ptr = cl::sycl::malloc_device<T>(size, queue);
    auto e = queue.memcpy(static_cast<void*>(ptr),
                          static_cast<const void*>(values), sizeof(T) * size);
    e.wait();
    return ptr;
  } else {
    return cl::sycl::buffer<T, 1>(values, cl::sycl::range<1>(size));
  }
}

// No-op, buffer is freed by end of scope
template <typename T>
void free(cl::sycl::buffer<T, 1> /*buffer*/, cl::sycl::queue& /*queue*/){};

template <typename T>
void free(T* ptr, cl::sycl::queue& queue) {
  cl::sycl::free(ptr, queue);
};

// No-op, buffer is freed by end of scope
template <typename T>
cl::sycl::event enqueue_free(cl::sycl::buffer<T, 1> /*buffer*/,
                             const std::vector<cl::sycl::event>& /*events*/,
                             cl::sycl::queue& queue) {
  // return dummy event
  return queue.submit([&](cl::sycl::handler& cgh) { cgh.host_task([=]() {}); });
};

template <typename T>
cl::sycl::event enqueue_free(T* ptr, const std::vector<cl::sycl::event>& events,
                             cl::sycl::queue& queue) {
  return queue.submit([=](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.host_task([=]() { cl::sycl::free(ptr, queue); });
  });
};

}  // namespace helpers
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_HELPERS_MEM_UTILS_H_
