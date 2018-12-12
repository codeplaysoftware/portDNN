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
#ifndef SYCLDNN_BENCH_FIXTURE_EIGEN_BACKEND_PROVIDER_H_
#define SYCLDNN_BENCH_FIXTURE_EIGEN_BACKEND_PROVIDER_H_

#include <cassert>

#include <unsupported/Eigen/CXX11/Tensor>
#include "sycldnn/backend/eigen_backend.h"

namespace sycldnn {
namespace bench {

template <typename Provider>
struct BackendProvider;

/**
 * Specialisation of the benchmark backend using Eigen.
 *
 * Provides access to sycldnn::backend::BackendProvider using Eigen and its
 * helper methods to allocate and deallocate memory.
 */
template <>
struct BackendProvider<sycldnn::backend::EigenBackend> {
 private:
  using EigenBackend = sycldnn::backend::EigenBackend;

  template <typename T>
  using Pointer = typename EigenBackend::pointer_type<T>;

 public:
  /**
   * Construct an BackendProvider using the static queue interface
   * managed by this provider.
   */
  BackendProvider() : device_{get_eigen_queue()}, backend_{device_} {}

  /** Get an EigenBackend instance.  */
  EigenBackend get_backend() { return backend_; }

  /** Allocate a SYCL buffer of type T for use with the EigenBackend. */
  template <typename T>
  Pointer<T> allocate(size_t size) {
    size_t bytes = size * sizeof(T);
    return static_cast<T*>(device_.allocate(bytes));
  }

  /** Deallocate a previously allocated pointer. */
  template <typename T>
  void deallocate(Pointer<T> ptr) {
    device_.deallocate(ptr);
  }

 private:
  /**
   * Keep a static instance of an Eigen::QueueInterface to prevent rebuilding
   * kernels for every benchmark instance.
   */
  Eigen::QueueInterface* get_eigen_queue() {
    static Eigen::QueueInterface queue_interface{cl::sycl::default_selector{}};
    return &queue_interface;
  }

  Eigen::SyclDevice device_;
  EigenBackend backend_;
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_FIXTURE_EIGEN_BACKEND_PROVIDER_H_
