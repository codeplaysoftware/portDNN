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

#define EIGEN_USE_SYCL
#include "unsupported/Eigen/CXX11/Tensor"

#include "test/backend/eigen_backend_test_fixture.h"
using EigenExternalDeathTest = EigenBackendTest;
using EigenInternalDeathTest = EigenBackendTest;

namespace {
cl::sycl::default_selector selector{};
}  // namespace
std::unique_ptr<Eigen::QueueInterface> EigenBackendTest::queue_interface_{
    new Eigen::QueueInterface{selector}};
Eigen::SyclDevice EigenBackendTest::device_{
    EigenBackendTest::queue_interface_.get()};
sycldnn::backend::EigenBackend EigenBackendTest::backend_{
    EigenBackendTest::device_};

TEST_F(EigenExternalDeathTest, FetchNonexistingBuffer) {
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr1 = backend_.allocate<float>(buffer_size);
  float* ptr2 = nullptr;
  ASSERT_DEATH(backend_.get_buffer(ptr2, n_elems),
               "The pointer is not registered in the map");
}
TEST_F(EigenExternalDeathTest, FetchBeforeAllocating) {
  float* ptr = nullptr;
  ASSERT_DEATH(backend_.get_buffer(ptr, 0), "There are no pointers allocated");
}
TEST_F(EigenExternalDeathTest, FetchAfterDeallocating) {
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr = backend_.allocate<float>(buffer_size);
  backend_.deallocate(ptr);
  ASSERT_DEATH(backend_.get_buffer(ptr, n_elems),
               "There are no pointers allocated");
}
TEST_F(EigenInternalDeathTest, FetchNonexistingBuffer) {
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr1 = backend_.allocate<float>(buffer_size);
  float* ptr2 = nullptr;
  ASSERT_DEATH(backend_.get_buffer_internal(ptr2, n_elems),
               "The pointer is not registered in the map");
}
TEST_F(EigenInternalDeathTest, FetchBeforeAllocating) {
  float* ptr = nullptr;
  ASSERT_DEATH(backend_.get_buffer_internal(ptr, 0),
               "There are no pointers allocated");
}
TEST_F(EigenInternalDeathTest, FetchAfterDeallocating) {
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr = backend_.allocate<float>(buffer_size);
  backend_.deallocate(ptr);
  ASSERT_DEATH(backend_.get_buffer_internal(ptr, n_elems),
               "There are no pointers allocated");
}
