/*
 * Copyright Codeplay Software Ltd.
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

#ifndef SYCLDNN_TEST_BATCHNORM_FIXTURE_H_
#define SYCLDNN_TEST_BATCHNORM_FIXTURE_H_

#include <gtest/gtest.h>

#if defined(SNN_TEST_SYCLBLAS)
#include "sycldnn/backend/sycl_blas_backend.h"
#else
#include "sycldnn/backend/snn_backend.h"
#endif

#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/launch.h"
#include "sycldnn/batchnorm/operation.h"
#include "sycldnn/batchnorm/params.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

#if defined(SNN_TEST_SYCLBLAS)
using Backend = sycldnn::backend::SyclBLASBackend;
#else
using Backend = sycldnn::backend::SNNBackend;
#endif

inline sycldnn::batchnorm::BatchNormParams getBatchNormParams(
    std::array<int, 4> in_shape, sycldnn::DataFormat data_format) {
  sycldnn::batchnorm::BatchNormParams params;
  params.batch = in_shape[0];
  params.rows = in_shape[1];
  params.cols = in_shape[2];
  params.channels = in_shape[3];
  params.input_format = data_format;
  return params;
}
template <typename DType, typename Direction, typename Operation>
struct BatchNormFixture;

template <typename DType, typename Direction>
struct BatchNormFixture<DType, Direction, sycldnn::batchnorm::Training>
    : public BackendTestFixture<Backend> {
  using DataType = DType;

  void test_batchnorm(std::vector<DataType> const& exp,
                      std::vector<DataType> const&,
                      std::vector<DataType> const&,
                      sycldnn::batchnorm::BatchNormParams const& params,
                      DataType max_val = static_cast<DataType>(0)) {
    auto input_size =
        params.batch * params.rows * params.cols * params.channels;
    ASSERT_EQ(input_size, exp.size());
    const auto size = exp.size();

    std::vector<DataType> input =
        iota_initialised_data<DataType>(input_size, max_val);

    std::vector<DataType> beta(params.channels, 0.f);
    std::vector<DataType> gamma(params.channels, 1.f);
    std::vector<DataType> mean(params.channels, 0.f);
    std::vector<DataType> variance(params.channels, 1.f);
    std::vector<DataType> output(size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input_size, input);
    auto beta_gpu =
        provider.get_initialised_device_memory(params.channels, beta);
    auto gamma_gpu =
        provider.get_initialised_device_memory(params.channels, gamma);
    auto mean_gpu =
        provider.get_initialised_device_memory(params.channels, mean);
    auto variance_gpu =
        provider.get_initialised_device_memory(params.channels, variance);
    auto out_gpu = provider.get_initialised_device_memory(size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(beta_gpu);
      provider.deallocate_ptr(gamma_gpu);
      provider.deallocate_ptr(mean_gpu);
      provider.deallocate_ptr(variance_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::batchnorm::launch<DataType, Backend,
                                             sycldnn::batchnorm::Training>(
        inp_gpu, beta_gpu, gamma_gpu, mean_gpu, variance_gpu, out_gpu, params,
        backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(size, out_gpu, output);

    for (size_t i = 0; i < size; i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(exp[i], output[i], 100u);
    }
  }
};

template <typename DType, typename Direction>
struct BatchNormFixture<DType, Direction, sycldnn::batchnorm::Inference>
    : public BackendTestFixture<Backend> {
  using DataType = DType;

  void test_batchnorm(std::vector<DataType> const& exp,
                      std::vector<DataType> const& exp_mean,
                      std::vector<DataType> const& exp_variance,
                      sycldnn::batchnorm::BatchNormParams const& params,
                      DataType max_val = static_cast<DataType>(0)) {
    auto input_size =
        params.batch * params.rows * params.cols * params.channels;
    ASSERT_EQ(input_size, exp.size());
    const auto size = exp.size();

    std::vector<DataType> input =
        iota_initialised_data<DataType>(input_size, max_val);

    std::vector<DataType> beta(params.channels, 0.f);
    std::vector<DataType> gamma(params.channels, 1.f);
    std::vector<DataType> output(size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input_size, input);
    auto beta_gpu =
        provider.get_initialised_device_memory(params.channels, beta);
    auto gamma_gpu =
        provider.get_initialised_device_memory(params.channels, gamma);
    auto mean_gpu =
        provider.get_initialised_device_memory(params.channels, exp_mean);
    auto variance_gpu =
        provider.get_initialised_device_memory(params.channels, exp_variance);
    auto out_gpu = provider.get_initialised_device_memory(size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(beta_gpu);
      provider.deallocate_ptr(gamma_gpu);
      provider.deallocate_ptr(mean_gpu);
      provider.deallocate_ptr(variance_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::batchnorm::launch<DataType, Backend,
                                             sycldnn::batchnorm::Inference>(
        inp_gpu, beta_gpu, gamma_gpu, mean_gpu, variance_gpu, out_gpu, params,
        backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(size, out_gpu, output);

    for (size_t i = 0; i < size; i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(exp[i], output[i], 100u);
    }
  }
};

#endif  // SYCLDNN_TEST_BATCHNORM_FIXTURE_H_
