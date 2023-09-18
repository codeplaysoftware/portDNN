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

#include <unsupported/Eigen/CXX11/Tensor>

#include "test/backend/backend_test_fixture.h"

#include "portdnn/backend/eigen_backend.h"

#include "src/backend/eigen_backend_provider.h"

#include <stddef.h>

#include <CL/sycl.hpp>

using EigenExternalHandlerTest =
    BackendTestFixture<sycldnn::backend::EigenBackend>;

TEST_F(EigenExternalHandlerTest, CheckQueue) {
  auto& provider = this->provider_;
  auto& backend = provider.get_backend();
  auto d_queue = provider.get_eigen_device().sycl_queue();
  auto b_queue = backend.get_queue();
  ASSERT_EQ(d_queue, b_queue);
}
TEST_F(EigenExternalHandlerTest, GetBufferExternalCheckSizes) {
  auto& provider = this->provider_;
  auto& backend = provider.get_backend();
  auto& device = provider.get_eigen_device();
  size_t buffer_size = 1024;
  size_t n_elems = buffer_size / sizeof(float);
  auto ptr = static_cast<float*>(device.allocate(buffer_size));
  auto mem_object = backend.get_mem_object(ptr, n_elems);
  auto backend_buffer = mem_object.get_buffer();
  EXPECT_EQ(buffer_size, backend_buffer.get_size());
}
TEST_F(EigenExternalHandlerTest, FillExternalBufferThenCheck) {
  using TensorType = Eigen::Tensor<float, 1>;
  using Tensor = Eigen::TensorMap<TensorType>;

  auto& provider = this->provider_;
  auto& device = provider.get_eigen_device();
  auto& backend = provider.get_backend();
  size_t n_floats = 16;
  size_t buffer_size = n_floats * sizeof(float);
  auto ptr = static_cast<float*>(device.allocate(buffer_size));

  Tensor tensor{ptr, n_floats};
  tensor.device(device) = tensor.constant(static_cast<float>(4));
  // Currently there is a problem with using async mode in Eigen, which requires
  // us to wait until the kernel is finished.
  // TODO(jwlawson): remove wait once no longer needed
  device.synchronize();

  // First check that the buffer returned by the Eigen has the correct contents.
  auto device_buffer = device.get_sycl_buffer(ptr);
  auto converted_buffer =
      device_buffer.reinterpret<float, 1>(cl::sycl::range<1>{n_floats});
  auto eigen_host_access =
      converted_buffer.get_access<cl::sycl::access::mode::read>();
  for (size_t i = 0; i < n_floats; ++i) {
    EXPECT_EQ(static_cast<float>(4), eigen_host_access[i]);
  }
  // Now check that the buffer returned by the Eigen backend has the correct
  // contents.
  auto mem_object = backend.get_mem_object(ptr, n_floats);
  auto backend_buffer = mem_object.get_buffer();
  auto snn_host_access =
      backend_buffer.get_access<cl::sycl::access::mode::read>();
  for (size_t i = 0; i < n_floats; ++i) {
    EXPECT_EQ(static_cast<float>(4), snn_host_access[i]);
  }
}
TEST_F(EigenExternalHandlerTest, ExternalPointerOffset) {
  auto& provider = this->provider_;
  auto& backend = provider.get_backend();
  auto& device = provider.get_eigen_device();
  size_t size = 1024;
  auto ptr1 = static_cast<int*>(device.allocate(size));
  auto ptr2 = ptr1 + 1;
  size_t exp1 = 1;
  auto mem_object_2 = backend.get_mem_object(ptr2, 1);
  EXPECT_EQ(exp1, mem_object_2.get_offset());
  auto ptr3 = ptr2 + 10;
  size_t exp2 = 11;
  auto mem_object_3 = backend.get_mem_object(ptr3, 1);
  EXPECT_EQ(exp2, mem_object_3.get_offset());
}
