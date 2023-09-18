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
#ifndef PORTDNN_SRC_BACKEND_SNN_BACKEND_PROVIDER_H_
#define PORTDNN_SRC_BACKEND_SNN_BACKEND_PROVIDER_H_

#include "portdnn/backend/snn_backend.h"
#include "portdnn/helpers/macros.h"

#include "src/backend/backend_provider.h"

namespace sycldnn {
namespace backend {

/** Specialisation of the backend provider for the SNNBackend.  */
template <>
struct BackendProvider<SNNBackend> {
 public:
  template <typename T>
  using Pointer = SNNBackend::pointer_type<T>;

  /** Default constructor using cached SYCL queue. */
  BackendProvider() : backend_{get_sycl_queue()} {}

  /** Disable copy constructors. */
  SNN_DISABLE_COPY(BackendProvider);

  /** Return this backend. */
  SNNBackend& get_backend() { return backend_; }

  /** Allocate memory on the device and initialise it with the provided data. */
  template <typename T>
  Pointer<T> get_initialised_device_memory(size_t size,
                                           std::vector<T> const& data) {
    if (!size) {
      return Pointer<T>{};
    }
    auto gpu_ptr = Pointer<T>{size};
    auto event = backend_.get_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc =
          gpu_ptr.get_buffer()
              .template get_access<cl::sycl::access::mode::discard_write>(
                  cgh, cl::sycl::range<1>{size},
                  cl::sycl::id<1>{gpu_ptr.get_offset()});
      cgh.copy(data.data(), acc);
    });
    event.wait_and_throw();
    return gpu_ptr;
  }

  /** Copy the device memory into the provided host vector. */
  template <typename T>
  void copy_device_data_to_host(size_t size, Pointer<T> gpu_ptr,
                                std::vector<T>& host_data) {
    host_data.resize(size);
    auto event = backend_.get_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc = gpu_ptr.get_buffer()
                     .template get_access<cl::sycl::access::mode::read>(
                         cgh, cl::sycl::range<1>{size},
                         cl::sycl::id<1>{gpu_ptr.get_offset()});
      cgh.copy(acc, host_data.data());
    });
    event.wait_and_throw();
  }

  /** Deallocate a device pointer. */
  template <typename T>
  void deallocate_ptr(Pointer<T> ptr) {
    SNN_UNUSED_VAR(ptr);
  }

 private:
  /** The backend that this provides. */
  SNNBackend backend_;

  /** Return a cached SYCL queue. */
  cl::sycl::queue& get_sycl_queue() {
    // Rethrow any SYCL exceptions as std::exceptions.
    auto exception_handler = [](cl::sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (cl::sycl::exception const& e) {
          throw std::runtime_error(e.what());
        }
      }
    };
    // By making the SYCL queue static any compiled kernels will be cached,
    // and so do not need to be recompiled for each test.
    static cl::sycl::queue queue{cl::sycl::default_selector{},
                                 exception_handler};
    return queue;
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_SRC_BACKEND_SNN_BACKEND_PROVIDER_H_
