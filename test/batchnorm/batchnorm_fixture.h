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

#ifndef PORTDNN_TEST_BATCHNORM_FIXTURE_H_
#define PORTDNN_TEST_BATCHNORM_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/batchnorm/direction.h"
#include "portdnn/batchnorm/launch.h"
#include "portdnn/batchnorm/params.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"
#include "test/helpers/transpose.h"

inline sycldnn::batchnorm::BatchNormParams getBatchNormParams(
    std::array<int, 4> in_shape, bool is_training, float momentum,
    float epsilon) {
  sycldnn::batchnorm::BatchNormParams params;
  params.batch = in_shape[0];
  params.rows = in_shape[1];
  params.cols = in_shape[2];
  params.channels = in_shape[3];
  params.is_training = is_training;
  params.momentum = momentum;
  params.epsilon = epsilon;
  params.input_format = sycldnn::DataFormat::NHWC;
  return params;
}

template <class T>
const std::vector<T>& transposeInput(
    sycldnn::batchnorm::BatchNormParams const& params,
    std::vector<T>& trInputData, const std::vector<T>& inputData) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    transpose(trInputData, inputData, params.batch, params.rows * params.cols,
              params.channels);
    return trInputData;
  }
  return inputData;
}

template <class T>
const std::vector<T>& transposeOutput(
    sycldnn::batchnorm::BatchNormParams const& params,
    std::vector<T>& trOutputData, const std::vector<T>& outputData) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    transpose(trOutputData, outputData, params.batch, params.channels,
              params.rows * params.cols);
    return trOutputData;
  }
  return outputData;
}

template <typename Triple, typename Direction>
struct BatchNormFixture;

template <typename Triple>
struct BatchNormFixture<Triple, sycldnn::batchnorm::Forward>
    : public BackendTestFixture<typename Triple::SecondType> {
  using DataType = typename Triple::FirstType;
  using Backend = typename Triple::SecondType;
  static constexpr sycldnn::DataFormat INPUT_FORMAT =
      Triple::ThirdType::input_layout;

  void test_batchnorm(std::vector<DataType> const& exp_running_mean,
                      std::vector<DataType> const& exp_running_var,
                      std::vector<DataType> const& exp_output,
                      sycldnn::batchnorm::BatchNormParams params,
                      DataType max_input_val, DataType max_beta_val,
                      DataType max_gamma_val, DataType max_input_mean_val,
                      DataType max_input_var_val) {
    ASSERT_EQ(params.input_format, sycldnn::DataFormat::NHWC)
        << "Tests should be written for the NHWC layout. The input layout is "
           "set from the fixture type.";
    params.input_format = INPUT_FORMAT;

    auto input_size =
        params.batch * params.rows * params.cols * params.channels;
    const auto size = exp_output.size();
    ASSERT_EQ(input_size, size);

    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(input_size, max_input_val);
    std::vector<DataType> beta =
        iota_initialised_data<DataType>(params.channels, max_beta_val);
    std::vector<DataType> gamma =
        iota_initialised_data<DataType>(params.channels, max_gamma_val);
    std::vector<DataType> input_mean =
        iota_initialised_data<DataType>(params.channels, max_input_mean_val);
    std::vector<DataType> input_var =
        iota_initialised_data<DataType>(params.channels, max_input_var_val);
    std::vector<DataType> outputData(size);
    std::vector<DataType>& output = outputData;

    std::vector<DataType> trInputData;
    const std::vector<DataType>& input =
        transposeInput(params, trInputData, inputData);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input_size, input);
    auto beta_gpu =
        provider.get_initialised_device_memory(params.channels, beta);
    auto gamma_gpu =
        provider.get_initialised_device_memory(params.channels, gamma);
    auto input_mean_gpu =
        provider.get_initialised_device_memory(params.channels, input_mean);
    auto input_variance_gpu =
        provider.get_initialised_device_memory(params.channels, input_var);
    auto running_mean_gpu =
        provider.get_initialised_device_memory(params.channels, input_mean);
    auto running_variance_gpu =
        provider.get_initialised_device_memory(params.channels, input_var);
    auto out_gpu = provider.get_initialised_device_memory(size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(beta_gpu);
      provider.deallocate_ptr(gamma_gpu);
      provider.deallocate_ptr(input_mean_gpu);
      provider.deallocate_ptr(input_variance_gpu);
      provider.deallocate_ptr(running_mean_gpu);
      provider.deallocate_ptr(running_variance_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::batchnorm::launch<DataType, Backend,
                                             sycldnn::batchnorm::Forward>(
        inp_gpu, beta_gpu, gamma_gpu, input_mean_gpu, input_variance_gpu,
        running_mean_gpu, running_variance_gpu, out_gpu, params, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    if (params.is_training) {
      provider.copy_device_data_to_host(params.channels, running_mean_gpu,
                                        output);

      for (size_t i = 0; i < static_cast<size_t>(params.channels); i++) {
        SCOPED_TRACE("Element: " + std::to_string(i));
        SNN_ALMOST_EQUAL_EPS(exp_running_mean[i], output[i], 10u, 1e-5);
      }

      provider.copy_device_data_to_host(params.channels, running_variance_gpu,
                                        output);

      for (size_t i = 0; i < static_cast<size_t>(params.channels); i++) {
        SCOPED_TRACE("Element: " + std::to_string(i));
        SNN_ALMOST_EQUAL_EPS(exp_running_var[i], output[i], 10u, 1e-5);
      }
    }

    provider.copy_device_data_to_host(size, out_gpu, outputData);
    std::vector<DataType> trOutputData;
    output = transposeOutput(params, trOutputData, outputData);

    for (size_t i = 0; i < size; i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL_EPS(exp_output[i], output[i], 10u, 2e-5);
    }
  }
};

template <typename Triple>
struct BatchNormFixture<Triple, sycldnn::batchnorm::Gradient>
    : public BackendTestFixture<typename Triple::SecondType> {
  using DataType = typename Triple::FirstType;
  using Backend = typename Triple::SecondType;
  static constexpr sycldnn::DataFormat INPUT_FORMAT =
      Triple::ThirdType::input_layout;

  void test_batchnorm(std::vector<DataType> const& exp_out_grad,
                      std::vector<DataType> const& exp_beta_grad,
                      std::vector<DataType> const& exp_gamma_grad,
                      sycldnn::batchnorm::BatchNormParams params,
                      DataType max_input_val, DataType max_gradient_val,
                      DataType max_gamma_val, DataType max_pop_mean_val,
                      DataType max_pop_var_val) {
    ASSERT_EQ(params.input_format, sycldnn::DataFormat::NHWC)
        << "Tests should be written for the NHWC layout. The input layout is "
           "set from the fixture type.";
    params.input_format = INPUT_FORMAT;

    auto input_size =
        params.batch * params.rows * params.cols * params.channels;
    const auto size = exp_out_grad.size();
    ASSERT_EQ(input_size, size);

    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(input_size, max_input_val);
    std::vector<DataType> gradientData =
        iota_initialised_data<DataType>(input_size, max_gradient_val);
    std::vector<DataType> gamma =
        iota_initialised_data<DataType>(params.channels, max_gamma_val);
    std::vector<DataType> pop_mean =
        iota_initialised_data<DataType>(params.channels, max_pop_mean_val);
    std::vector<DataType> pop_var =
        iota_initialised_data<DataType>(params.channels, max_pop_var_val);
    std::vector<DataType> beta_grad(params.channels);
    std::vector<DataType> gamma_grad(params.channels);
    std::vector<DataType> outputData(size);
    std::vector<DataType>& output = outputData;

    std::vector<DataType> trInputData;
    const std::vector<DataType>& input =
        transposeInput(params, trInputData, inputData);

    std::vector<DataType> trGradientData;
    const std::vector<DataType>& gradient =
        transposeInput(params, trGradientData, gradientData);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input_size, input);
    auto gradient_gpu =
        provider.get_initialised_device_memory(input_size, gradient);
    auto gamma_gpu =
        provider.get_initialised_device_memory(params.channels, gamma);
    auto pop_mean_gpu =
        provider.get_initialised_device_memory(params.channels, pop_mean);
    auto pop_variance_gpu =
        provider.get_initialised_device_memory(params.channels, pop_var);
    auto beta_grad_gpu =
        provider.get_initialised_device_memory(params.channels, beta_grad);
    auto gamma_grad_gpu =
        provider.get_initialised_device_memory(params.channels, gamma_grad);
    auto out_gpu = provider.get_initialised_device_memory(size, output);

    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(gradient_gpu);
      provider.deallocate_ptr(gamma_gpu);
      provider.deallocate_ptr(pop_mean_gpu);
      provider.deallocate_ptr(pop_variance_gpu);
      provider.deallocate_ptr(beta_grad_gpu);
      provider.deallocate_ptr(gamma_grad_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::batchnorm::launch<DataType, Backend,
                                             sycldnn::batchnorm::Gradient>(
        inp_gpu, gradient_gpu, gamma_gpu, pop_mean_gpu, pop_variance_gpu,
        beta_grad_gpu, gamma_grad_gpu, out_gpu, params, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(params.channels, beta_grad_gpu, output);

    for (size_t i = 0; i < (size_t)params.channels; i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL_EPS(exp_beta_grad[i], output[i], 10u, 1e-5);
    }

    provider.copy_device_data_to_host(params.channels, gamma_grad_gpu, output);

    for (size_t i = 0; i < (size_t)params.channels; i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL_EPS(exp_gamma_grad[i], output[i], 30u, 1e-2);
    }

    provider.copy_device_data_to_host(size, out_gpu, outputData);
    std::vector<DataType> trOutputData;
    output = transposeOutput(params, trOutputData, outputData);

    for (size_t i = 0; i < size; i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL_EPS(exp_out_grad[i], output[i], 30u, 1e-2);
    }
  }
};

#endif  // PORTDNN_TEST_BATCHNORM_FIXTURE_H_
