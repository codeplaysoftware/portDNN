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

#ifndef SYCLDNN_TEST_POINTWISE_POINTWISE_FIXTURE_H_
#define SYCLDNN_TEST_POINTWISE_POINTWISE_FIXTURE_H_

#include <gtest/gtest.h>

#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sycldnn/backend/eigen_backend.h"

#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/pointwise/operators.h"

#include "test/gen/eigen_generated_test_fixture.h"
#include "test/gen/iota_initialised_data.h"

#include <vector>

template <typename DType, template <typename> class Op, typename Direction>
struct PointwiseFixture
    : public EigenGeneratedTestFixture<DType, sycldnn::backend::EigenBackend> {
  using DataType = DType;

  void test_pointwise(const std::vector<DataType>& exp) {
    const auto size = exp.size();

    std::vector<DataType> input = iota_initialised_signed_data<DataType>(size);
    std::vector<DataType> output(size);

    auto inp_gpu = this->get_initialised_device_memory(size, input);
    auto out_gpu = this->get_initialised_device_memory(size, output);

    auto status = sycldnn::pointwise::launch<DataType, Op, Direction>(
        inp_gpu, out_gpu, size, this->backend_);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait();

    this->copy_device_data_to_host(size, out_gpu, output);
    this->deallocate_ptr(inp_gpu);
    this->deallocate_ptr(out_gpu);

    for (size_t i = 0; i < size; ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      if (std::is_same<DataType, double>::value) {
        EXPECT_DOUBLE_EQ(exp[i], output[i]);
      } else {
        EXPECT_FLOAT_EQ(exp[i], output[i]);
      }
    }
  }
};

template <typename DType, template <typename T> class Op>
struct PointwiseFixture<DType, Op, sycldnn::pointwise::Gradient>
    : public EigenGeneratedTestFixture<DType, sycldnn::backend::EigenBackend> {
  using DataType = DType;

  void test_pointwise(const std::vector<DataType>& exp) {
    constexpr DataType tolerance = 0.00001;

    const auto size = exp.size();

    std::vector<DataType> input_forward =
        iota_initialised_signed_data<DataType>(size);
    std::vector<DataType> output_forward(size);
    std::vector<DataType> input_backprop =
        iota_initialised_signed_data<DataType>(size);
    std::vector<DataType> output_backprop(size);

    auto inp_fwd_gpu = this->get_initialised_device_memory(size, input_forward);
    auto out_fwd_gpu =
        this->get_initialised_device_memory(size, output_forward);

    auto fwd_status =
        sycldnn::pointwise::launch<DataType, Op, sycldnn::pointwise::Forward>(
            inp_fwd_gpu, out_fwd_gpu, size, this->backend_);
    ASSERT_EQ(sycldnn::StatusCode::OK, fwd_status.status);

    auto inp_bk_gpu = this->get_initialised_device_memory(size, input_backprop);
    auto out_bk_gpu =
        this->get_initialised_device_memory(size, output_backprop);

    auto bk_status =
        sycldnn::pointwise::launch<DataType, Op, sycldnn::pointwise::Gradient>(
            out_fwd_gpu, inp_bk_gpu, out_bk_gpu, size, this->backend_);
    ASSERT_EQ(sycldnn::StatusCode::OK, bk_status.status);

    fwd_status.event.wait();
    bk_status.event.wait();

    this->copy_device_data_to_host(size, out_bk_gpu, output_backprop);
    this->deallocate_ptr(inp_fwd_gpu);
    this->deallocate_ptr(out_fwd_gpu);
    this->deallocate_ptr(inp_bk_gpu);
    this->deallocate_ptr(out_bk_gpu);

    for (size_t i = 0; i < size; ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      ASSERT_NEAR(exp[i], output_backprop[i], tolerance);
    }
  }
};
#endif  // SYCLDNN_TEST_POINTWISE_POINTWISE_FIXTURE_H_
