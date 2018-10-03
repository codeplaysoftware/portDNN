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
#ifndef SYCLDNN_TEST_GEN_EIGEN_GENERATED_TEST_FIXTURE_H_
#define SYCLDNN_TEST_GEN_EIGEN_GENERATED_TEST_FIXTURE_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

// TODO(jwlawson): remove cassert when no longer needed before Eigen include
#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>

#include "test/backend/eigen_backend_test_fixture.h"

template <typename DataType, typename Backend>
struct EigenGeneratedTestFixture : public EigenBackendTest<Backend> {
 protected:
  /** Allocate memory on the device and initialise it with the provided data. */
  DataType* get_initialised_device_memory(size_t size,
                                          std::vector<DataType> const& data) {
    auto device = this->get_eigen_device();
    size_t n_bytes = size * sizeof(DataType);
    DataType* gpu_ptr = static_cast<DataType*>(device.allocate(n_bytes));
    device.memcpyHostToDevice(gpu_ptr, data.data(), n_bytes);
    return gpu_ptr;
  }
  /** Copy the device memory into the provided host vector. */
  void copy_device_data_to_host(size_t size, DataType* gpu_ptr,
                                std::vector<DataType>& host_data) {
    host_data.resize(size);
    auto device = this->get_eigen_device();
    size_t n_bytes = size * sizeof(DataType);
    device.memcpyDeviceToHost(host_data.data(), gpu_ptr, n_bytes);
  }
  /** Deallocate a device pointer. */
  void deallocate_ptr(DataType* ptr) {
    auto device = this->get_eigen_device();
    device.deallocate(ptr);
  }
};

#endif  // SYCLDNN_TEST_GEN_EIGEN_GENERATED_TEST_FIXTURE_H_
