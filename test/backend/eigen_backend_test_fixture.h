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
#ifndef SYCLDNN_TEST_BACKEND_EIGEN_BACKEND_TEST_FIXTURE_H_
#define SYCLDNN_TEST_BACKEND_EIGEN_BACKEND_TEST_FIXTURE_H_
#include <gtest/gtest.h>

// Need to ensure that Eigen is included before the backend.
#include <unsupported/Eigen/CXX11/Tensor>
#include "sycldnn/backend/eigen_backend.h"

struct EigenBackendTest : public ::testing::Test {
 protected:
  virtual void TearDown() { device_.deallocate_all(); }

  // By making the Eigen device static any compiled kernels will be cached, and
  // so do not need to be recompiled for each test.
  static std::unique_ptr<Eigen::QueueInterface> queue_interface_;
  static Eigen::SyclDevice device_;
  static sycldnn::backend::EigenBackend backend_;
};
#endif  // SYCLDNN_TEST_BACKEND_EIGEN_BACKEND_TEST_FIXTURE_H_
