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
#include <gtest/gtest.h>

#ifndef SNN_TEST_EIGEN
#error This Eigen specific test cannot compile without Eigen
#endif

#include "portdnn/backend/eigen_backend.h"
#include "src/backend/eigen_backend_provider.h"

#include "test/backend/backend_test_fixture.h"

#include <stddef.h>
#include <string>

using Backends = ::testing::Types<sycldnn::backend::EigenBackend>;

template <typename Backend>
using ExternalDeathTest = BackendTestFixture<Backend>;
template <typename Backend>
using InternalDeathTest = BackendTestFixture<Backend>;
TYPED_TEST_SUITE(ExternalDeathTest, Backends);
TYPED_TEST_SUITE(InternalDeathTest, Backends);

TYPED_TEST(ExternalDeathTest, FetchNonexistingBuffer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto& backend = this->provider_.get_backend();
  float* ptr1 = backend.template allocate<float>(n_elems);
  ASSERT_NE(nullptr, ptr1);
  float* ptr2 = nullptr;
  ASSERT_ANY_THROW(backend.get_mem_object(ptr2, n_elems));
}
TYPED_TEST(ExternalDeathTest, FetchBeforeAllocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  float* ptr = nullptr;
  ASSERT_ANY_THROW(this->provider_.get_backend().get_mem_object(ptr, 0));
}
TYPED_TEST(ExternalDeathTest, FetchAfterDeallocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto& backend = this->provider_.get_backend();
  float* ptr = backend.template allocate<float>(n_elems);
  backend.deallocate(ptr);
  ASSERT_ANY_THROW(backend.get_mem_object(ptr, n_elems));
}
TYPED_TEST(InternalDeathTest, FetchNonexistingBuffer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto& backend = this->provider_.get_backend();
  float* ptr1 = backend.template allocate<float>(buffer_size);
  ASSERT_NE(nullptr, ptr1);
  float* ptr2 = nullptr;
  ASSERT_ANY_THROW(backend.get_mem_object_internal(ptr2, n_elems));
}
TYPED_TEST(InternalDeathTest, FetchBeforeAllocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  float* ptr = nullptr;
  ASSERT_ANY_THROW(
      this->provider_.get_backend().get_mem_object_internal(ptr, 0));
}
TYPED_TEST(InternalDeathTest, FetchAfterDeallocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto& backend = this->provider_.get_backend();
  float* ptr = backend.template allocate<float>(n_elems);
  backend.deallocate(ptr);
  ASSERT_ANY_THROW(backend.get_mem_object_internal(ptr, n_elems));
}
