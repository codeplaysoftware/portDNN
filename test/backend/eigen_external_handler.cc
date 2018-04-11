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

namespace {
cl::sycl::default_selector selector{};
}  // namespace
std::unique_ptr<Eigen::QueueInterface> EigenBackendTest::queue_interface_{
    new Eigen::QueueInterface{selector}};
Eigen::SyclDevice EigenBackendTest::device_{
    EigenBackendTest::queue_interface_.get()};
sycldnn::backend::EigenBackend EigenBackendTest::backend_{
    EigenBackendTest::device_};

TEST_F(EigenBackendTest, CheckQueue) {
  auto d_queue = device_.sycl_queue();
  auto b_queue = backend_.get_queue();
  ASSERT_EQ(d_queue, b_queue);
}
TEST_F(EigenBackendTest, GetBufferExternalCheckSizes) {
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  float* ptr = static_cast<float*>(device_.allocate(buffer_size));
  auto backend_buffer = backend_.get_buffer(ptr, n_elems);
  EXPECT_EQ(buffer_size, backend_buffer.get_size());
}
TEST_F(EigenBackendTest, FillExternalBufferThenCheck) {
  using TensorType = Eigen::Tensor<float, 1>;
  using Tensor = Eigen::TensorMap<TensorType>;

  size_t n_floats = 16;
  size_t buffer_size = n_floats * sizeof(float);
  float* ptr = static_cast<float*>(device_.allocate(buffer_size));

  Tensor tensor{ptr, n_floats};
  tensor.device(device_) = tensor.constant(static_cast<float>(4));
  // First check that the buffer returned by the Eigen has the correct contents.
  auto device_buffer = device_.get_sycl_buffer(ptr);
  {
    // This is required for ComputeCpp 0.6, to ensure that the host accessors
    // used below can access the data.
    auto workaround_buffer =
        device_buffer.get_access<cl::sycl::access::mode::read>();
  }
  auto converted_buffer = device_buffer.reinterpret<float, 1>(
      cl::sycl::range<1>{static_cast<size_t>(n_floats)});
  auto eigen_host_access =
      converted_buffer.get_access<cl::sycl::access::mode::read>();
  for (size_t i = 0; i < n_floats; ++i) {
    EXPECT_EQ(static_cast<float>(4), eigen_host_access[i]);
  }
  // Now check that the buffer returned by the Eigen backend has the correct
  // contents.
  auto backend_buffer = backend_.get_buffer(ptr, n_floats);
  auto snn_host_access =
      backend_buffer.get_access<cl::sycl::access::mode::read>();
  for (size_t i = 0; i < n_floats; ++i) {
    EXPECT_EQ(static_cast<float>(4), snn_host_access[i]);
  }
}
TEST_F(EigenBackendTest, ExternalPointerOffset) {
  size_t size = 1024;
  int* ptr1 = static_cast<int*>(device_.allocate(size));
  int* ptr2 = ptr1 + 1;
  size_t exp1 = 1;
  EXPECT_EQ(exp1, backend_.get_offset(ptr2));
  int* ptr3 = ptr2 + 10;
  size_t exp2 = 11;
  EXPECT_EQ(exp2, backend_.get_offset(ptr3));
}
