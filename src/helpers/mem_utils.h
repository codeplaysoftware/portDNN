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

namespace sycldnn {
namespace helpers {

template <typename T, bool IsUSM>
class mem_utils {
 public:
  mem_utils(cl::sycl::queue& queue) : queue_(queue){};

  cl::sycl::buffer<T, 1> alloc(size_t size) const {
    return cl::sycl::buffer<T, 1>(cl::sycl::range<1>(size));
  }

  // No-op, buffer is freed by end of scope
  void free(cl::sycl::buffer<T, 1> /*buffer*/) const {};

  // No-op, buffer is freed by end of scope
  cl::sycl::event enqueue_free(cl::sycl::buffer<T, 1> /*buffer*/,
                               const std::vector<cl::sycl::event>& /*events*/) {
    // return dummy event
    return queue_.submit(
        [&](cl::sycl::handler& cgh) { cgh.host_task([=]() {}); });
  };

 private:
  cl::sycl::queue queue_;
};

template <typename T>
class mem_utils<T, true> {
 public:
  mem_utils(cl::sycl::queue& queue) : queue_(queue){};

  T* alloc(size_t size) const {
    return cl::sycl::malloc_device<T>(size, queue_);
  }

  void free(T* ptr) const { cl::sycl::free(ptr, queue_); };

  cl::sycl::event enqueue_free(T* ptr,
                               const std::vector<cl::sycl::event>& events) {
    return queue_.submit([&](cl::sycl::handler& cgh) {
      cgh.depends_on(events);
      cgh.host_task([=]() { cl::sycl::free(ptr, queue_); });
    });
  };

 private:
  cl::sycl::queue queue_;
};

}  // namespace helpers
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_HELPERS_MEM_UTILS_H_
