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
#include <gtest/gtest.h>

// TODO(jwlawson): remove cassert when no longer needed before Eigen include
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>

#ifdef SNN_TEST_SYCLBLAS_MATMULS
#include "src/backend/syclblas_backend_provider.h"
#include "sycldnn/backend/sycl_blas_backend.h"
#endif

#include "src/backend/eigen_backend_provider.h"
#include "sycldnn/backend/eigen_backend.h"
#include "test/backend/backend_test_fixture.h"

#ifdef EIGEN_EXCEPTIONS
#define MAYBE_DEATH(a, b) ASSERT_ANY_THROW(a)
#else
#define MAYBE_DEATH(a, b) ASSERT_DEATH(a, b)
#endif

using Backends = ::testing::Types<
#ifdef SNN_TEST_SYCLBLAS_MATMULS
    sycldnn::backend::SyclBLASBackend,
#endif
    sycldnn::backend::EigenBackend>;

template <typename Backend>
using ExternalDeathTest = BackendTestFixture<Backend>;
template <typename Backend>
using InternalDeathTest = BackendTestFixture<Backend>;
TYPED_TEST_CASE(ExternalDeathTest, Backends);
TYPED_TEST_CASE(InternalDeathTest, Backends);

TYPED_TEST(ExternalDeathTest, FetchNonexistingBuffer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto backend = this->provider_.get_backend();
  float* ptr1 = backend.template allocate<float>(n_elems);
  ASSERT_NE(nullptr, ptr1);
  float* ptr2 = nullptr;
  MAYBE_DEATH(backend.get_buffer(ptr2, n_elems), "Cannot access null pointer");
}
TYPED_TEST(ExternalDeathTest, FetchBeforeAllocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  float* ptr = nullptr;
  MAYBE_DEATH(this->provider_.get_backend().get_buffer(ptr, 0),
              "There are no pointers allocated");
}
TYPED_TEST(ExternalDeathTest, FetchAfterDeallocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto backend = this->provider_.get_backend();
  float* ptr = backend.template allocate<float>(n_elems);
  backend.deallocate(ptr);
  MAYBE_DEATH(backend.get_buffer(ptr, n_elems),
              "There are no pointers allocated");
}
TYPED_TEST(InternalDeathTest, FetchNonexistingBuffer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto backend = this->provider_.get_backend();
  float* ptr1 = backend.template allocate<float>(buffer_size);
  ASSERT_NE(nullptr, ptr1);
  float* ptr2 = nullptr;
  MAYBE_DEATH(backend.get_buffer_internal(ptr2, n_elems),
              "Cannot access null pointer");
}
TYPED_TEST(InternalDeathTest, FetchBeforeAllocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  float* ptr = nullptr;
  MAYBE_DEATH(this->provider_.get_backend().get_buffer_internal(ptr, 0),
              "There are no pointers allocated");
}
TYPED_TEST(InternalDeathTest, FetchAfterDeallocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto backend = this->provider_.get_backend();
  float* ptr = backend.template allocate<float>(n_elems);
  backend.deallocate(ptr);
  MAYBE_DEATH(backend.get_buffer_internal(ptr, n_elems),
              "There are no pointers allocated");
}
