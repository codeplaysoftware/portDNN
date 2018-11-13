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
#ifndef SYCLDNN_TEST_BACKEND_BACKEND_TEST_FIXTURE_H_
#define SYCLDNN_TEST_BACKEND_BACKEND_TEST_FIXTURE_H_
#include <gtest/gtest.h>

// TODO(jwlawson): remove cassert when no longer needed before Eigen include
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sycldnn/backend/eigen_backend.h"

#ifdef SNN_TEST_SYCLBLAS_MATMULS
#include "sycldnn/backend/sycl_blas_backend.h"
#endif  // SNN_TEST_SYCLBLAS_MATMULS

template <typename Backend>
struct BackendTest;

template <typename Backend>
struct BackendTest : public ::testing::Test {
  BackendTest() : backend_{get_eigen_device()} {}

 protected:
  virtual void TearDown() {
    auto& device = get_eigen_device();
    device.sycl_queue().wait_and_throw();
    device.deallocate_all();
  }

  Eigen::SyclDevice& get_eigen_device() {
    // By making the Eigen device static any compiled kernels will be cached,
    // and so do not need to be recompiled for each test.
    static Eigen::QueueInterface queue_interface{cl::sycl::default_selector{}};
    static Eigen::SyclDevice device{&queue_interface};
    return device;
  }
  Backend backend_;
};

#ifdef SNN_TEST_SYCLBLAS_MATMULS
template <>
struct BackendTest<sycldnn::backend::SyclBLASBackend> : public ::testing::Test {
  BackendTest() : backend_{get_default_queue()} {}

 protected:
  virtual void TearDown() { get_default_queue().wait_and_throw(); }

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

  sycldnn::backend::SyclBLASBackend backend_;
};
#endif  // SNN_TEST_SYCLBLAS_MATMULS
#endif  // SYCLDNN_TEST_BACKEND_BACKEND_TEST_FIXTURE_H_
