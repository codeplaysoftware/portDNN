/*
 * Copyright 2019 Codeplay Software Ltd
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
#ifndef SYCLDNN_SRC_BACKEND_SYCLBLAS_BACKEND_PROVIDER_H_
#define SYCLDNN_SRC_BACKEND_SYCLBLAS_BACKEND_PROVIDER_H_

#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/helpers/macros.h"

#include "src/backend/backend_provider.h"

namespace sycldnn {
namespace backend {
/**
 * Specialisation of the backend provider using SYCL-BLAS.
 *
 * Provides access to sycldnn::backend::BackendProvider using SYCL-BLAS and its
 * helper methods to allocate and deallocate memory.
 */
template <>
struct BackendProvider<SyclBLASBackend> {
  BackendProvider() : backend_{get_default_queue()} {}
  SNN_DISABLE_COPY(BackendProvider);

  template <typename T>
  using Pointer = SyclBLASBackend::pointer_type<T>;

  /** Return this backend. */
  SyclBLASBackend& get_backend() { return backend_; }
  /** Allocate memory on the device and initialise it with the provided data. */
  template <typename T>
  Pointer<T> get_initialised_device_memory(size_t size,
                                           std::vector<T> const& data) {
    auto executor = backend_.get_executor().get_policy_handler();
    auto gpu_ptr = executor.template allocate<T>(size);
    auto event = executor.copy_to_device(data.data(), gpu_ptr, size);
    event.back().wait_and_throw();
    return gpu_ptr;
  }
  /** Copy the device memory into the provided host vector. */
  template <typename T>
  void copy_device_data_to_host(size_t size, Pointer<T> gpu_ptr,
                                std::vector<T>& host_data) {
    host_data.resize(size);
    auto event = backend_.get_executor().get_policy_handler().copy_to_host(
        gpu_ptr, host_data.data(), size);
    event.wait_and_throw();
  }
  /** Deallocate a device pointer. */
  template <typename T>
  void deallocate_ptr(Pointer<T> ptr) {
    backend_.get_executor().get_policy_handler().deallocate(ptr);
  }

  /**
   * Keep a static instance of an cl::sycl::queue to prevent rebuilding
   * kernels for every benchmark instance.
   */
  cl::sycl::queue get_default_queue() {
    auto exception_handler = [](cl::sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (cl::sycl::exception const& e) {
          std::cerr << "Caught asynchronous SYCL exception:\n"
                    << e.what() << std::endl;
          throw;
        }
      }
    };
    static cl::sycl::queue queue{cl::sycl::default_selector(),
                                 exception_handler};
    return queue;
  }

 private:
  SyclBLASBackend backend_;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BACKEND_SYCLBLAS_BACKEND_PROVIDER_H_
