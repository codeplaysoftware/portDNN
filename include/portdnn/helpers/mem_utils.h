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
#ifndef INCLUDE_PORTDNN_HELPERS_MEM_UTILS_H_
#define INCLUDE_PORTDNN_HELPERS_MEM_UTILS_H_

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

template <typename T>
cl::sycl::event cpy(USMMemObject<T const> in_mem, USMMemObject<T> out_mem,
                    cl::sycl::queue& queue,
                    const std::vector<cl::sycl::event>& events) {
  // Fill output buffer with input data
  return queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto in = in_mem.read_mem(cgh).get_pointer();
    auto out = out_mem.write_mem(cgh).get_pointer();

    assert(in_mem.get_extent() == out_mem.get_extent());

    // TODO: make copy when ComputeCpp implements it
    cgh.memcpy(static_cast<void*>(out), static_cast<const void*>(in),
               in_mem.get_extent() * sizeof(T));
  });
};

template <typename T>
cl::sycl::event cpy(BufferMemObject<T const> in_mem, BufferMemObject<T> out_mem,
                    cl::sycl::queue& queue,
                    const std::vector<cl::sycl::event>& events) {
  return queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto in_acc = in_mem.read_accessor(cgh).get_accessor();
    auto out_acc = out_mem.write_accessor(cgh).get_accessor();

    cgh.copy(in_acc, out_acc);
  });
};

template <typename T>
inline __attribute__((always_inline)) void free_ptr(
    const cl::sycl::queue& queue, T* ptr) {
  cl::sycl::free(ptr, queue);
}

template <typename T, typename... Args>
inline __attribute__((always_inline)) void free_ptr(
    const cl::sycl::queue& queue, T* ptr, Args... ptrs) {
  cl::sycl::free(ptr, queue);
  free_ptr(queue, ptrs...);
}

template <typename... Args>
cl::sycl::event enqueue_free(cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events,
                             Args... ptrs) {
  return queue.submit([=](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.host_task([=]() { free_ptr(queue, ptrs...); });
  });
}

template <typename T>
cl::sycl::event enqueue_free(cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events,
                             cl::sycl::buffer<T, 1>& buffer) {
  SNN_UNUSED_VAR(buffer);
  return queue.submit([=](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.host_task([=]() {});
  });
}

template <typename T, typename... Args>
cl::sycl::event enqueue_free(cl::sycl::queue& queue,
                             const std::vector<cl::sycl::event>& events,
                             cl::sycl::buffer<T, 1>& buffer,
                             Args... /* buffers */) {
  SNN_UNUSED_VAR(buffer);
  return queue.submit([=](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.host_task([=]() {});
  });
}

}  // namespace helpers
}  // namespace sycldnn
#endif  // INCLUDE_PORTDNN_HELPERS_MEM_UTILS_H_
