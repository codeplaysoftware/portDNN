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
#include "test/backend/eigen_backend_test_fixture.h"

#ifdef EIGEN_EXCEPTIONS
#define MAYBE_DEATH(a, b) ASSERT_ANY_THROW(a)
#else
#define MAYBE_DEATH(a, b) ASSERT_DEATH(a, b)
#endif

using EigenExternalDeathTest = EigenBackendTest;
using EigenInternalDeathTest = EigenBackendTest;

TEST_F(EigenExternalDeathTest, FetchNonexistingBuffer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr1 = backend_.allocate<float>(buffer_size);
  ASSERT_NE(nullptr, ptr1);
  float* ptr2 = nullptr;
  MAYBE_DEATH(backend_.get_buffer(ptr2, n_elems), "Cannot access null pointer");
}
TEST_F(EigenExternalDeathTest, FetchBeforeAllocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  float* ptr = nullptr;
  MAYBE_DEATH(backend_.get_buffer(ptr, 0), "There are no pointers allocated");
}
TEST_F(EigenExternalDeathTest, FetchAfterDeallocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr = backend_.allocate<float>(buffer_size);
  backend_.deallocate(ptr);
  MAYBE_DEATH(backend_.get_buffer(ptr, n_elems),
              "There are no pointers allocated");
}
TEST_F(EigenInternalDeathTest, FetchNonexistingBuffer) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr1 = backend_.allocate<float>(buffer_size);
  ASSERT_NE(nullptr, ptr1);
  float* ptr2 = nullptr;
  MAYBE_DEATH(backend_.get_buffer_internal(ptr2, n_elems),
              "Cannot access null pointer");
}
TEST_F(EigenInternalDeathTest, FetchBeforeAllocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  float* ptr = nullptr;
  MAYBE_DEATH(backend_.get_buffer_internal(ptr, 0),
              "There are no pointers allocated");
}
TEST_F(EigenInternalDeathTest, FetchAfterDeallocating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr = backend_.allocate<float>(buffer_size);
  backend_.deallocate(ptr);
  MAYBE_DEATH(backend_.get_buffer_internal(ptr, n_elems),
              "There are no pointers allocated");
}
