/*
 * Copyright 2018 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#ifndef SYCLDNN_BENCH_FIXTURE_SYCLBLAS_BACKEND_PROVIDER_H_
#define SYCLDNN_BENCH_FIXTURE_SYCLBLAS_BACKEND_PROVIDER_H_

#include "sycldnn/backend/sycl_blas_backend.h"

namespace sycldnn {
namespace bench {

template <typename Provider>
struct BackendProvider;
/**
 * Specialisation of the benchmark backend using SYCL-BLAS.
 *
 * Provides access to sycldnn::backend::BackendProvider using SYCL-BLAS and its
 * helper methods to allocate and deallocate memory.
 */
template <>
struct BackendProvider<sycldnn::backend::SyclBLASBackend> {
 private:
  using SyclBLASBackend = sycldnn::backend::SyclBLASBackend;

  /** The pointer type which can be allocated for the Backend. */
  template <typename T>
  using Pointer = typename SyclBLASBackend::pointer_type<T>;

 public:
  /**
   * Construct an BackendProvider using the static queue interface
   * managed by this provider.
   */
  BackendProvider() : backend_{get_default_queue()} {}

  /** Get an SyclBLASBackend instance. */
  SyclBLASBackend& get_backend() { return backend_; }

  /** Allocate a SYCL buffer of type T for use with the SyclBLASBackend. */
  template <typename T>
  Pointer<T> allocate(size_t size) {
    size_t bytes = size * sizeof(T);
    return backend_.allocate<T>(bytes);
  }

  /** Deallocate a previously allocated pointer. */
  template <typename T>
  void deallocate(Pointer<T> ptr) {
    backend_.deallocate(ptr);
  }

 private:
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

  SyclBLASBackend backend_;
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_FIXTURE_SYCLBLAS_BACKEND_PROVIDER_H_
